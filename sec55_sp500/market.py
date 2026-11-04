"""Paper Section 5.5 market reconstruction, independent of ML frameworks.

Numerical conventions are documented in PROTOCOL.md. Units inside
GARCH are percentage log returns; prices and payoffs are index points.
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln, ndtr

ROOT = Path(__file__).resolve().parents[1]


def window_data(window_id):
    with (ROOT/'data/processed/paper_rolling_windows_300.csv').open() as f:
        windows = list(csv.DictReader(f))
    w = windows[window_id]
    with (ROOT/'data/processed/sp500_prices_20070612_20180102.csv').open() as f:
        rows = [r for r in csv.DictReader(f) if w['price_start'] <= r['Date'] <= w['price_end']]
    assert len(rows) == 1218
    prices = np.array([float(r['Adj.Close']) for r in rows])
    return w, prices, np.log(prices[1:]/prices[:-1])


def filter_calibration(x, p):
    mu, omega, alpha, gamma, beta, nu = p
    eps = x-mu
    first = float(np.var(x))
    forcing = omega+(alpha+gamma*(eps[:-1] < 0))*eps[:-1]**2
    rest, _ = lfilter([1.], [1., -beta], forcing, zi=[beta*first])
    return eps, np.r_[first, rest]


def fit_garch(calibration_returns):
    """Conditional MLE, variance-standardized Student-t (declared convention)."""
    x = np.asarray(calibration_returns)*100
    variance = float(np.var(x))

    def nll(p):
        eps, h = filter_calibration(x, p)
        if np.any(h <= 0) or not np.isfinite(h).all(): return 1e30
        nu = p[-1]
        logpdf = (gammaln((nu+1)/2)-gammaln(nu/2)
                  -.5*np.log((nu-2)*np.pi*h)
                  -(nu+1)/2*np.log1p(eps**2/((nu-2)*h)))
        return float(-logpdf.sum())

    fits = []
    for a, g, b, nu in [(.05,.1,.85,8.),(.05,.15,.8,5.),(.1,.05,.8,15.)]:
        result = minimize(nll, [np.mean(x),variance*(1-a-g/2-b),a,g,b,nu],
                          method='SLSQP',
                          bounds=[(-5*np.sqrt(variance),5*np.sqrt(variance)),(1e-9,100*variance),
                                  (0,.999),(0,1.998),(0,.999),(2.05,100)],
                          constraints=[{'type':'ineq','fun':lambda p:.999-p[2]-p[3]/2-p[4]}],
                          options={'maxiter':1000,'ftol':1e-9})
        fits.append(dict(success=bool(result.success),message=str(result.message),
                         nll=float(result.fun),parameters=result.x.tolist(),iterations=int(result.nit)))
    valid = [r for r in fits if r['success'] and r['parameters'][2]+r['parameters'][3]/2+r['parameters'][4] <= .9990001]
    if not valid: raise RuntimeError('No converged feasible GJR-GARCH fit')
    best = min(valid,key=lambda r:r['nll'])
    p = np.array(best['parameters'])
    eps, h = filter_calibration(x,p)
    state = dict(h0=float(h[-1]),epsilon0=float(eps[-1]))
    state['h1'] = float(p[1]+(p[2]+p[3]*(eps[-1]<0))*eps[-1]**2+p[4]*h[-1])
    return p, state, dict(parameters=p.tolist(),parameter_names=['mu_percent','omega_percent_squared','alpha','gamma','beta','nu'],
                         initial_state=state,initial_sigma0_daily=float(np.sqrt(h[-1])/100),
                         calibration_observations=len(x),fits=fits,
                         student_t='variance-standardized; explicit local convention, not specified by paper')


def paths(p,state,s0,n,rng,days=30):
    """At k use h[k] = Var(epsilon[k]|past), not the h[k+1] forecast.

    Day zero stores the last calibration volatility sigma0. First simulate h1
    from h0/epsilon0; then draw epsilon1. All other dates follow the same order.
    """
    mu,omega,a,g,b,nu = p
    prices=np.empty((n,days+1));vol=np.empty_like(prices)
    prices[:,0]=s0
    h=np.full(n,state['h0']);eps=np.full(n,state['epsilon0'])
    vol[:,0]=np.sqrt(h)/100
    for k in range(1,days+1):
        h=omega+(a+g*(eps<0))*eps**2+b*h
        z=rng.standard_t(nu,n)*np.sqrt((nu-2)/nu)
        eps=np.sqrt(h)*z
        prices[:,k]=prices[:,k-1]*np.exp((mu+eps)/100)
        vol[:,k]=np.sqrt(h)/100
    if not np.isfinite(prices).all(): raise RuntimeError('Nonfinite price simulation; no clipping permitted')
    return prices,vol


def observed_path(p,state,s0,returns):
    mu,omega,a,g,b,nu = p
    prices=np.r_[s0,s0*np.exp(np.cumsum(returns))][None,:]
    vol=np.empty_like(prices);h=state['h0'];eps=state['epsilon0']
    vol[0,0]=np.sqrt(h)/100
    for k,r in enumerate(returns,start=1):
        h=omega+(a+g*(eps<0))*eps**2+b*h
        vol[0,k]=np.sqrt(h)/100
        eps=100*r-mu
    return prices,vol


def bs_premium(s0,sigma0,days=30):
    v=sigma0*np.sqrt(days)
    return float(s0*(2*ndtr(v/2)-1))


def bs_hedge(prices,vol,strike,hedge_days):
    days=np.array(hedge_days)
    s=prices[:,days];v=vol[:,days]*np.sqrt(30-days)
    return ndtr((np.log(s/strike)+.5*v*v)/v)


def sample_tensors(prices,strike,hedge_days):
    hp=prices[:,list(hedge_days)+[30]]
    return (np.log(hp[:,:-1]).astype('float32'),np.diff(hp,axis=1).astype('float32'),
            np.maximum(prices[:,-1]-strike,0).astype('float32'))


def cvar(loss,alpha):
    x=np.sort(np.asarray(loss,dtype=float))[::-1]
    mass=len(x)*(1-alpha);whole=int(np.floor(mass));fraction=mass-whole
    return float((x[:whole].sum()+(fraction*x[whole] if fraction>1e-10 else 0))/mass)


def pnl(prices,holdings,premium,strike,hedge_days):
    hp=prices[:,list(hedge_days)+[30]]
    return premium-np.maximum(prices[:,-1]-strike,0)+(holdings*np.diff(hp,axis=1)).sum(1)


def summarize(prices,holdings,premium,strike,hedge_days):
    pl=pnl(prices,holdings,premium,strike,hedge_days);loss=-pl
    turnover=np.abs(np.diff(np.c_[np.zeros(len(pl)),holdings,np.zeros(len(pl))],axis=1)).sum(1)
    return dict(samples=len(pl),mean_pnl=float(pl.mean()),mse=float(np.mean(pl**2)),
                probability_loss=float(np.mean(pl<0)),cvar75=cvar(loss,.75),cvar95=cvar(loss,.95),
                mean_turnover=float(turnover.mean()),max_absolute_holdings=float(np.abs(holdings).max()),
                minimum_pnl=float(pl.min()),maximum_pnl=float(pl.max()))


def check_market():
    w,prices,r=window_data(0)
    assert len(r[:1187])==1187 and len(r[1187:])==30
    p,state,fit=fit_garch(r[:1187]);s0=prices[1187]
    observed,vol=observed_path(p,state,s0,r[1187:])
    assert np.allclose(observed[0],prices[1187:])
    prefix,prefixvol=observed_path(p,state,s0,r[1187:1190])
    assert np.allclose(prefixvol,vol[:,:4])
    simulated,sv=paths(p,state,s0,32,np.random.default_rng(4))
    assert np.allclose(sv[:,0],np.sqrt(state['h0'])/100)
    assert np.allclose(sv[:,1],np.sqrt(state['h1'])/100)
    assert np.allclose(sv[:,0],vol[:,0])
    grid=np.arange(31,dtype=float)[None,:]+100
    assert np.allclose(pnl(grid,np.ones((1,6)),7,100,[0,5,10,15,20,25]),7)
    assert np.isclose(cvar([1,2,3,4],.625),(4+.5*3)/1.5)
    return dict(checks='PASS: sigma0/sigma1 alignment, simulation recursion, history alignment, causal filtering, CVaR and P&L',
                initial_sigma0=fit['initial_sigma0_daily'],next_sigma1=float(np.sqrt(state['h1'])/100),
                premium_sigma0=bs_premium(s0,fit['initial_sigma0_daily']),fit=fit)


if __name__=='__main__':
    print(json.dumps(check_market(),indent=2))
