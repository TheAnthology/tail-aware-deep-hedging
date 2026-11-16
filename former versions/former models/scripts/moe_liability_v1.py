"""Causal current Heston call liability for the frozen MoE experiment.

Reuse the existing QuantLib price surfaces read-only. Only current spot,
variance and remaining time enter a value; neither realized payoff nor future
states are read. Cached cubic price interpolation is a numerical convention,
validated separately against direct QuantLib. No pricing-grid derivative is
needed. Out-of-support or invalid-bound interpolations use direct pricing.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from heston52_benchmark import HestonGridHedge

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/preference_moe_v1/liability'


class HestonLiability:
    def __init__(self, cache_dir=None):
        self.cache_dir = Path(cache_dir) if cache_dir is not None else ROOT/'results/section52_quantlib_cache'
        self.grid = HestonGridHedge(cache_dir=self.cache_dir, sensitive_variance=.001)
        self.q = self.grid.q0
        self.total_points = 0
        self.direct_fallback_points = 0
        self.boundary_fallback_points = 0
        self.cache_paths = {}
        for days in range(1,31):
            path = self.grid._cache_path(days)
            if path is None or not path.exists():
                raise FileNotFoundError(f'Frozen price cache missing; do not silently rebuild: {path}')
            self.cache_paths[days] = path
            self.grid.build(days)

    def values(self, spots, variances, remaining_days):
        """Vectorized prices at current observable states; integer days/365."""
        s,v = np.broadcast_arrays(np.asarray(spots,dtype=float), np.asarray(variances,dtype=float))
        if np.any(s<=0) or np.any(v<0) or not np.isfinite(s).all() or not np.isfinite(v).all():
            raise ValueError('Finite positive spot and nonnegative variance required')
        days = int(remaining_days)
        if days != remaining_days or not 0<=days<=30:
            raise ValueError('Integer remaining_days in[0,30] required')
        if days==0:
            return np.maximum(s-self.grid.p.strike,0.)
        fs,fv = s.ravel(),v.ravel()
        inside = self.grid._inside(fs,fv)
        result = np.empty(len(fs),dtype='float64')
        if inside.any():
            result[inside] = self.grid.splines[days].ev(
                np.log(fs[inside]/self.grid.p.strike),
                np.log1p(fv[inside]/self.grid.variance_scale))
        # Cubic interpolation can overshoot the intrinsic bound near the
        # attainable V=0 boundary. Reject it and evaluate the actual price;
        # never clip prices, spot, variance or surplus.
        lower = np.maximum(fs-self.grid.p.strike,0.)
        tolerance = 1e-10*np.maximum(np.maximum(fs,self.grid.p.strike),1.)
        invalid = np.zeros(len(fs),dtype=bool)
        invalid[inside] = (~np.isfinite(result[inside])
                           |(result[inside]<lower[inside]-tolerance[inside])
                           |(result[inside]>fs[inside]+tolerance[inside]))
        fallback = ~inside | invalid
        if fallback.any():
            ref = self.grid.reference
            with ref.active_date():
                for i in np.flatnonzero(fallback):
                    result[i] = ref._price(float(fs[i]),float(fv[i]),days)
        # Exact standard initial state, not a rounded premium or spline q.
        initial = (days==30)&(fs==100.)&(fv==.04)
        result[initial] = self.q
        self.total_points += len(fs)
        self.direct_fallback_points += int(fallback.sum())
        self.boundary_fallback_points += int(invalid.sum())
        assert np.isfinite(result).all()
        return result.reshape(s.shape)

    def price(self, market):
        """[paths,30] current call liability at the thirty decision dates."""
        s = np.asarray(market['spot']);v = np.asarray(market['variance'])
        if s.shape!=v.shape or s.ndim!=2 or s.shape[1]!=31:
            raise ValueError('Aligned spot/variance arrays [paths,31] required')
        # Explicit indexing makes this independent of terminal/future states.
        return np.stack([self.values(s[:,k],v[:,k],30-k) for k in range(30)],axis=1)


def validate():
    from heston52_market import sample_paths
    OUT.mkdir(parents=True,exist_ok=True)
    start = time.monotonic();provider = HestonLiability()
    source_hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in provider.cache_paths.values()}
    rng = np.random.default_rng(6239)
    market = sample_paths(256,rng)
    prices = provider.price(market)
    assert prices.shape==(256,30)
    assert np.array_equal(prices[:,0],np.full(256,provider.q))
    state = rng.integers(0,256,size=72); dates = rng.integers(0,30,size=72)
    records=[]
    for path,k in zip(state,dates):
        s=float(market['spot'][path,k]);v=float(market['variance'][path,k]);days=30-int(k)
        direct=provider.grid.reference.price(s,v,days)
        approx=float(provider.values(s,v,days))
        records.append({'source':'simulated','spot':s,'variance':v,'days':days,
                        'direct':direct,'interpolated':approx,'absolute_error':abs(approx-direct)})
    for days in [30,15,5,1]:
        center=provider.grid.p.rho/provider.grid.p.vol_of_vol*(provider.grid.p.kappa*provider.grid.p.theta*days/365.)
        for x,v in [(center+offset,0.) for offset in [-.002,-.001,0.,.001,.002]] + [(0.,1e-15),(0.,.04),(0.,.5)]:
            s=100*np.exp(x);direct=provider.grid.reference.price(s,v,days)
            approx=float(provider.values(s,v,days))
            records.append({'source':'boundary_stress','spot':s,'variance':v,'days':days,
                            'direct':direct,'interpolated':approx,'absolute_error':abs(approx-direct)})
    # Perturb all future fields after a date; values through that date remain
    # bitwise equal. In particular current C15 cannot use the terminal payoff.
    altered={key:value.copy() for key,value in market.items()}
    altered['spot'][:,16:] *= 1.7
    altered['variance'][:,16:] += .13
    altered['payoff']=np.full(256,999999.)
    changed=provider.price(altered)
    assert np.array_equal(prices[:,:16],changed[:,:16])
    terminal=provider.values(market['spot'][:,-1],market['variance'][:,-1],0)
    assert np.array_equal(terminal,np.maximum(market['spot'][:,-1]-100,0))
    errors=np.array([r['absolute_error'] for r in records]);types=np.array([r['source'] for r in records])
    summary={}
    for source in ['simulated','boundary_stress']:
        e=errors[types==source]
        summary[source]={'points':len(e),'max_absolute_error':float(e.max()),
                         'mean_absolute_error':float(e.mean()),'p95_absolute_error':float(np.quantile(e,.95))}
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==s for p,s in source_hashes.items())
    report={'q':provider.q,'checks':{'C0_exact_premium':True,'maturity_exact_payoff':True,
            'future_states_and_payoff_cannot_change_past_liability':True,'all_finite':True,
            'existing_caches_unchanged':True},'errors':summary,'validation_points':records,
            'read_only_cache_hashes':source_hashes,'elapsed_seconds':time.monotonic()-start,
            'provider_total_points':provider.total_points,
            'direct_fallback_points':provider.direct_fallback_points,
            'invalid_bound_fallback_points':provider.boundary_fallback_points,
            'method':'Read-only existing QuantLib cubic price splines; direct price outside grid or after bound violation. No derivative sensitive-state fallback required for value-only output; measured error explicitly reported.',
            'boundary':'Existing valuation-only QuantLib limit V<=1e-12 ->1e-12; actual market and Gate variance never changed.',
            'cache_builds':0,'additional_cache_bytes':0}
    (OUT/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (OUT/'REPORT.md').write_text('# Causal Heston option liability\n\n'
        'API: `HestonLiability().price(market)` returns `[N,30]`; `values(S,V,days)` prices individual observable states. '
        'No payoff or future-state access. The existing thirty price grids are read-only and no new cache is built. '
        f'Initial liability equals exact q={provider.q:.15f}; at expiry value equals realized call payoff.\n\n'
        +json.dumps(summary,indent=2)+'\n\n'
        'Validation perturbs every state after date15 and replaces payoff: current/past liabilities remain bitwise unchanged. '
        'Interpolation is approximate; errors above are direct QuantLib comparisons at72 simulated and32 boundary/stress states. '
        'The price bound fallback rejects spline overshoot rather than clipping. The existing valuation-only V<=1e-12 limit is retained.\n')
    print(json.dumps({k:report[k] for k in ['q','checks','errors','elapsed_seconds','direct_fallback_points','invalid_bound_fallback_points']},indent=2))

if __name__=='__main__':validate()
