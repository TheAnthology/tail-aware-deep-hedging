"""QuantLib reference and explicitly approximate cached model hedge for §5.2.

The paper states that QuantLib supplies the model hedge, but does not disclose
its integration order, Greek differencing, or interpolation. This module keeps
those numerical conventions explicit. `QuantLibReference` prices directly and
differences the price in spot/variance. `HestonGridHedge` samples QuantLib prices
on an optional cache grid and differentiates their spline. The latter is NOT a
claim of per-path QuantLib evaluation. Grid error must be checked before use.

Market variance is never changed in a simulated path. Since QuantLib's
HestonModel is numerically unstable at some extremely small positive inputs,
valuation at 0<=v<=epsilon uses a recorded positive-epsilon limit (1e-12).
The same effective variance is used for finite-difference Greek bumps. Negative
variance inputs are errors. Outside a grid, direct QuantLib Greeks are used;
there is no clipping or spline extrapolation.

Primary source for QuantLib model parameter order and analytic integration:
https://github.com/lballabio/QuantLib/blob/v1.40/ql/models/equity/hestonmodel.cpp
https://github.com/lballabio/QuantLib/blob/v1.40/ql/pricingengines/vanilla/analytichestonengine.hpp
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import RectBivariateSpline


@dataclass(frozen=True)
class HestonParameters:
    kappa: float = 1.0
    theta: float = 0.04
    vol_of_vol: float = 2.0
    rho: float = -0.7
    strike: float = 100.0
    horizon_days: int = 30
    year_days: int = 365


class QuantLibReference:
    """Direct analytic Heston pricing plus finite-difference Eq.(33) holdings.

    Mutable QuantLib pricing objects are reused for efficiency. This object is
    serial, not thread-safe. Its evaluation-date context restores global state.
    """
    def __init__(self, parameters: HestonParameters | None = None,
                 integration_order: int = 144, zero_variance_epsilon: float = 1e-12,
                 integration_mode: str = 'adaptive', relative_tolerance: float = 1e-8,
                 max_evaluations: int = 10000):
        import QuantLib as ql
        self.ql = ql
        self.p = parameters or HestonParameters()
        self.integration_order = int(integration_order)
        self.integration_mode = integration_mode
        self.relative_tolerance = float(relative_tolerance)
        self.max_evaluations = int(max_evaluations)
        self.zero_variance_epsilon = float(zero_variance_epsilon)
        if not 0 < self.zero_variance_epsilon <= 1e-6:
            raise ValueError('Use an explicitly small positive variance-limit epsilon')
        self.base_date = ql.Date(1,1,2020)
        self.spot_quote = ql.SimpleQuote(100.)
        curve = ql.YieldTermStructureHandle(ql.FlatForward(self.base_date,0.,ql.Actual365Fixed()))
        process = ql.HestonProcess(curve,curve,ql.QuoteHandle(self.spot_quote),.04,
                                  self.p.kappa,self.p.theta,self.p.vol_of_vol,self.p.rho)
        self.model = ql.HestonModel(process)
        if integration_mode == 'adaptive':
            self.engine = ql.AnalyticHestonEngine(self.model,self.relative_tolerance,self.max_evaluations)
        elif integration_mode == 'laguerre':
            self.engine = ql.AnalyticHestonEngine(self.model,self.integration_order)
        elif integration_mode == 'asymptotic_cf':
            self.engine = ql.AnalyticHestonEngine(self.model,ql.AnalyticHestonEngine.AsymptoticChF,
                                                ql.AnalyticHestonEngine_Integration.gaussLaguerre(self.integration_order))
        else:
            raise ValueError('integration_mode must be adaptive, laguerre or asymptotic_cf')
        self.options = {}
        self._last_variance = .04
        self.direct_price_calls = 0
        self.zero_boundary_price_calls = 0

    def effective_variance(self,variance: float) -> float:
        """Numerical valuation boundary limit, never a market-path mutation."""
        v=float(variance)
        if not np.isfinite(v) or v<0:
            raise ValueError('Nonnegative finite variance required')
        if v<=self.zero_variance_epsilon:
            self.zero_boundary_price_calls+=1
            return self.zero_variance_epsilon
        return v

    def _check_price_bounds(self,value: float,spot: float,context: str):
        # No clipping: reject numerical failures beyond integration roundoff.
        tolerance=1e-7*max(1.,spot,self.p.strike)
        lower=max(spot-self.p.strike,0.)
        if not np.isfinite(value) or value<lower-tolerance or value>spot+tolerance:
            raise RuntimeError(f'QuantLib call violates no-arbitrage bounds ({context}): value={value}, spot={spot}, lower={lower}')

    @staticmethod
    def _check_holdings(delta,swap,context):
        if (not np.isfinite(delta) or not np.isfinite(swap)
                or delta<-.01 or delta>1.01 or abs(swap)>=1e8):
            raise RuntimeError(f'Numerically invalid Heston hedge ({context}): stock={delta}, swap={swap}')

    @contextmanager
    def active_date(self):
        settings = self.ql.Settings.instance()
        old = settings.evaluationDate
        settings.evaluationDate = self.base_date
        try: yield
        finally: settings.evaluationDate = old

    def _price(self, spot: float, variance: float, remaining_days: int) -> float:
        if not np.isfinite(spot) or spot <= 0 or not np.isfinite(variance) or variance < 0:
            raise ValueError('Positive finite spot and nonnegative finite variance required')
        if remaining_days <= 0:
            return float(max(spot-self.p.strike,0.))
        if int(remaining_days) != remaining_days:
            raise ValueError('Grid is keyed by integer remaining calendar days / 365')
        v = self.effective_variance(variance)
        self.spot_quote.setValue(float(spot))
        if v != self._last_variance:
            # QuantLib order is theta,kappa,sigma,rho,v0 (not constructor order).
            self.model.setParams(self.ql.Array([self.p.theta,self.p.kappa,self.p.vol_of_vol,self.p.rho,v]))
            self._last_variance = v
        days = int(remaining_days)
        if days not in self.options:
            option = self.ql.VanillaOption(self.ql.PlainVanillaPayoff(self.ql.Option.Call,self.p.strike),
                                         self.ql.EuropeanExercise(self.base_date+days))
            option.setPricingEngine(self.engine)
            self.options[days] = option
        result = float(self.options[days].NPV())
        self.direct_price_calls += 1
        self._check_price_bounds(result,float(spot),f'v={v}, days={days}')
        return result

    def price(self, spot: float, variance: float, remaining_days: int) -> float:
        with self.active_date():
            return self._price(float(spot),float(variance),remaining_days)

    def _greeks(self, spot: float, variance: float, remaining_days: int,
                spot_relative_bump: float, variance_absolute_bump: float,
                variance_relative_bump: float):
        s=float(spot)
        v=self.effective_variance(variance)
        if remaining_days <= 0:
            raise ValueError('No hedge is evaluated at expiry; L_v would be zero')
        h_s = max(1e-5,spot_relative_bump*s)
        h_v = max(variance_absolute_bump,variance_relative_bump*v)
        value = self._price(s,v,remaining_days)
        delta = (self._price(s+h_s,v,remaining_days)-self._price(s-h_s,v,remaining_days))/(2*h_s)
        if v >= h_v:
            dvar = (self._price(s,v+h_v,remaining_days)-self._price(s,v-h_v,remaining_days))/(2*h_v)
        else:
            # Second-order forward derivative at or near the v=0 boundary.
            dvar = (-3*value+4*self._price(s,v+h_v,remaining_days)-self._price(s,v+2*h_v,remaining_days))/(2*h_v)
        lv = -np.expm1(-self.p.kappa*remaining_days/self.p.year_days)/self.p.kappa
        swap=float(dvar/lv)
        self._check_holdings(float(delta),swap,f's={s}, effective_v={v}, days={remaining_days}')
        return value,float(delta),float(dvar),swap

    def greeks(self, spot: float, variance: float, remaining_days: int,
               spot_relative_bump: float = 1e-5, variance_absolute_bump: float = 1e-7,
               variance_relative_bump: float = 1e-3):
        with self.active_date():
            return self._greeks(spot,variance,remaining_days,spot_relative_bump,
                                variance_absolute_bump,variance_relative_bump)

    def holdings(self, spots, variances, remaining_days: int, **bump_kwargs):
        s,v = np.broadcast_arrays(np.asarray(spots,dtype=float),np.asarray(variances,dtype=float))
        result = np.empty(s.shape+(2,))
        with self.active_date():
            for i,(spot,var) in enumerate(zip(s.ravel(),v.ravel())):
                _,delta,_,swap = self._greeks(float(spot),float(var),remaining_days,
                                             bump_kwargs.get('spot_relative_bump',1e-5),
                                             bump_kwargs.get('variance_absolute_bump',1e-7),
                                             bump_kwargs.get('variance_relative_bump',1e-3))
                result.reshape(-1,2)[i] = delta,swap
        return result


def default_log_spot_grid():
    """Dense around the ATM boundary, coarser away from it; no extrapolation."""
    return np.unique(np.r_[np.linspace(-.8,-.1,20),np.linspace(-.1,-.02,25),
                           np.linspace(-.02,.02,161),np.linspace(.02,.1,25),
                           np.linspace(.1,.8,20)])


class HestonGridHedge:
    """QuantLib-price spline cache, explicit interpolation and direct fallback.

    Prices are represented in x=log(S/K), y=log1p(V/v_scale). The model hedge is
    dU/dS=(dU/dx)/S and dU/dV=(dU/dy)/(v_scale+V). This differentiates price, not
    volatility, and divides the second holding by the exact L_v from Eq.(33).
    """
    def __init__(self, reference: QuantLibReference | None = None,
                 x_nodes=None, variance_nodes=None, variance_scale: float = 1e-4,
                 cache_dir: str | Path | None = None,
                 sensitive_variance: float = 1e-4,
                 sensitive_log_width: float = .001):
        self.reference = reference or QuantLibReference()
        self.p = self.reference.p
        self.variance_scale = float(variance_scale)
        self.sensitive_variance = float(sensitive_variance)
        self.sensitive_log_width = float(sensitive_log_width)
        self.x = np.asarray(default_log_spot_grid() if x_nodes is None else x_nodes,dtype=float)
        self.v = np.asarray(np.unique(np.r_[np.linspace(0,1e-4,21),np.expm1(np.linspace(0,np.log1p(2./variance_scale),101))*variance_scale])
                            if variance_nodes is None else variance_nodes,dtype=float)
        if min(len(self.x),len(self.v)) < 4 or np.any(np.diff(self.x)<=0) or np.any(np.diff(self.v)<=0) or self.v[0]<0:
            raise ValueError('At least four strictly increasing nodes per axis required; V>=0')
        self.y = np.log1p(self.v/self.variance_scale)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:self.cache_dir.mkdir(parents=True,exist_ok=True)
        self.splines = {}
        self.fallback_points = 0
        self.sensitive_fallback_points = 0
        self.total_points = 0
        self.build_seconds = {}

    def configuration(self):
        return dict(method='QuantLib analytic prices -> RectBivariateSpline derivatives; direct QuantLib outside support',
                    parameters=asdict(self.p),quantlib_version=self.reference.ql.__version__,
                    integration_order=self.reference.integration_order,
                    integration_mode=self.reference.integration_mode,
                    relative_tolerance=self.reference.relative_tolerance,
                    max_evaluations=self.reference.max_evaluations,
                    zero_variance_epsilon=self.reference.zero_variance_epsilon,
                    boundary_policy='For valuation and Greek bumps only, every 0<=V<=epsilon uses effective V=epsilon; simulated market V is untouched',
                    numerical_guards='Reject call outside intrinsic..spot beyond 1e-7*max(spot,strike,1); stock hedge outside[-.01,1.01] or abs(swap)>=1e8; never clip',
                    sensitive_variance=self.sensitive_variance,
                    sensitive_log_width=self.sensitive_log_width,
                    sensitive_rule='direct QuantLib when V<=threshold and abs(log(S/K)-rho/sigma*(V+kappa*theta*tau))<=log_width',
                    variance_scale=self.variance_scale,x_nodes=self.x.tolist(),variance_nodes=self.v.tolist())

    def _cache_path(self, days):
        if self.cache_dir is None:return None
        config=self.configuration()
        # These new runtime-only safeguards do not alter old price-grid nodes:
        # default nodes are exactly zero or >=5e-6, and zero already used eps.
        # Preserve legacy cache identity when no positive node is in the newly
        # treated tiny-variance range. Any genuinely affected node invalidates.
        if not np.any((self.v>0)&(self.v<=self.reference.zero_variance_epsilon)):
            config.pop('boundary_policy')
        config.pop('numerical_guards')
        key=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
        return self.cache_dir/f'ql_prices_{days:02d}_{key}.npz'

    def build(self, remaining_days: int):
        days=int(remaining_days)
        if days<1 or days!=remaining_days:raise ValueError('Positive integer hedge horizon required')
        if days in self.splines:return
        start=time.monotonic();path=self._cache_path(days)
        if path is not None and path.exists():
            with np.load(path) as stored:values=stored['prices']
            if values.shape!=(len(self.x),len(self.y)):raise ValueError('Corrupt QuantLib cache shape')
        else:
            values=np.empty((len(self.x),len(self.y)))
            with self.reference.active_date():
                # Variance outer loop avoids repeating setParams at each stock.
                for j,v in enumerate(self.v):
                    for i,x in enumerate(self.x):
                        values[i,j]=self.reference._price(self.p.strike*np.exp(x),float(v),days)
            if path is not None:np.savez_compressed(path,prices=values)
        spots=self.p.strike*np.exp(self.x)
        tolerance=1e-7*np.maximum(np.maximum(spots,self.p.strike),1.)
        lower=np.maximum(spots-self.p.strike,0.)
        if (not np.isfinite(values).all() or np.any(values<lower[:,None]-tolerance[:,None])
                or np.any(values>spots[:,None]+tolerance[:,None])):
            raise RuntimeError(f'QuantLib price grid violates no-arbitrage bounds, remaining_days={days}')
        self.splines[days]=RectBivariateSpline(self.x,self.y,values,kx=3,ky=3,s=0)
        self.build_seconds[days]=time.monotonic()-start

    def _inside(self,s,v):
        x=np.log(s/self.p.strike)
        return (x>=self.x[0])&(x<=self.x[-1])&(v>=self.v[0])&(v<=self.v[-1])

    def _sensitive(self,s,v,remaining_days):
        # The low-variance high-frequency characteristic-function phase locates
        # a very sharp transition. A price spline's differentiated approximation
        # was not accurate there; explicitly use direct QuantLib finite Greeks.
        tau=remaining_days/self.p.year_days
        center=self.p.rho/self.p.vol_of_vol*(v+self.p.kappa*self.p.theta*tau)
        return (v<=self.sensitive_variance)&(np.abs(np.log(s/self.p.strike)-center)<=self.sensitive_log_width*1.001)

    def price_and_holdings(self, spots, variances, remaining_days: int):
        s,v=np.broadcast_arrays(np.asarray(spots,dtype=float),np.asarray(variances,dtype=float))
        if np.any(s<=0) or np.any(v<0) or not np.isfinite(s).all() or not np.isfinite(v).all():
            raise ValueError('Finite positive spots and nonnegative variances required')
        self.build(remaining_days)
        flat_s,flat_v=s.ravel(),v.ravel()
        sensitive=self._sensitive(flat_s,flat_v,remaining_days)
        inside=self._inside(flat_s,flat_v)&~sensitive
        values=np.empty(len(flat_s));holding=np.empty((len(flat_s),2))
        if inside.any():
            si,vi=flat_s[inside],flat_v[inside]
            x=np.log(si/self.p.strike);y=np.log1p(vi/self.variance_scale)
            spline=self.splines[int(remaining_days)]
            values[inside]=spline.ev(x,y)
            holding[inside,0]=spline.ev(x,y,dx=1,dy=0)/si
            dvar=spline.ev(x,y,dx=0,dy=1)/(self.variance_scale+vi)
            lv=-np.expm1(-self.p.kappa*remaining_days/self.p.year_days)/self.p.kappa
            holding[inside,1]=dvar/lv
        if (~inside).any():
            with self.reference.active_date():
                for i in np.flatnonzero(~inside):
                    price,delta,_,swap=self.reference._greeks(float(flat_s[i]),float(flat_v[i]),int(remaining_days),1e-5,1e-7,1e-3)
                    values[i]=price;holding[i]=delta,swap
            self.fallback_points+=int((~inside).sum())
            self.sensitive_fallback_points+=int(sensitive.sum())
        self.total_points+=len(flat_s)
        if (not np.isfinite(holding).all() or np.any(holding[:,0]<-.01)
                or np.any(holding[:,0]>1.01) or np.any(np.abs(holding[:,1])>=1e8)):
            bad=(~np.isfinite(holding).all(1))|(holding[:,0]<-.01)|(holding[:,0]>1.01)|(np.abs(holding[:,1])>=1e8)
            i=int(np.flatnonzero(bad)[0])
            raise RuntimeError(f'Invalid interpolated/direct hedge: s={flat_s[i]}, v={flat_v[i]}, days={remaining_days}, holdings={holding[i].tolist()}')
        return values.reshape(s.shape),holding.reshape(s.shape+(2,))

    def holdings(self,spots,variances,remaining_days: int):
        return self.price_and_holdings(spots,variances,remaining_days)[1]

    def path_holdings(self, spot_paths, variance_paths):
        spots=np.asarray(spot_paths);variances=np.asarray(variance_paths)
        if spots.shape!=variances.shape or spots.ndim!=2 or spots.shape[1]!=31:
            raise ValueError('Expect aligned arrays (paths,31)')
        result=np.empty((len(spots),30,2))
        for k in range(30):result[:,k,:]=self.holdings(spots[:,k],variances[:,k],30-k)
        return result

    @property
    def q0(self):
        return self.reference.price(100.,.04,30)

    def hedge_paths(self,data):
        return self.path_holdings(data['spot'],data['variance'])

    def validation_report(self,**kwargs):
        return self.validate(**kwargs)

    def validate(self,remaining_days=(30,15,1),points_per_day=64,seed=7101):
        """Off-grid direct QuantLib check. Report errors; do not auto-accept."""
        rng=np.random.default_rng(seed);reports=[]
        for days in remaining_days:
            x=rng.uniform(max(self.x[0],-.1),min(self.x[-1],.1),points_per_day)
            v=np.expm1(rng.uniform(self.y[0],min(self.y[-1],np.log1p(.5/self.variance_scale)),points_per_day))*self.variance_scale
            # Include near-zero-variance ATM states important under Feller breach.
            x=np.r_[x,[0.,-1e-4,1e-4,0.]];v=np.r_[v,[0.,0.,1e-10,.04]]
            center=self.p.rho/self.p.vol_of_vol*self.p.kappa*self.p.theta*days/self.p.year_days
            offsets=np.array([-.002,-.001,-.0005,-.0002,0.,.0002,.0005,.001,.002])
            x=np.r_[x,center+offsets];v=np.r_[v,np.zeros(len(offsets))]
            s=self.p.strike*np.exp(x)
            price,holding=self.price_and_holdings(s,v,int(days))
            direct=np.array([self.reference.greeks(float(si),float(vi),int(days)) for si,vi in zip(s,v)])
            price_error=np.abs(price-direct[:,0]);stock_error=np.abs(holding[:,0]-direct[:,1]);swap_error=np.abs(holding[:,1]-direct[:,3])
            reports.append(dict(remaining_days=int(days),points=len(s),price_abs_max=float(price_error.max()),
                                price_abs_mean=float(price_error.mean()),stock_holding_abs_max=float(stock_error.max()),
                                stock_holding_abs_mean=float(stock_error.mean()),swap_holding_abs_max=float(swap_error.max()),
                                swap_holding_abs_mean=float(swap_error.mean())))
        return dict(configuration=self.configuration(),checks=reports,
                    fallback_points=self.fallback_points,sensitive_fallback_points=self.sensitive_fallback_points,
                    note='Interpolation and finite-difference errors reported, not assumed negligible. Refine nodes/bump/order and compare P&L before accepting.')


if __name__=='__main__':
    reference=QuantLibReference()
    print(json.dumps(dict(quantlib_version=reference.ql.__version__,
                          initial_price_and_greeks=reference.greeks(100.,.04,30)),indent=2))
