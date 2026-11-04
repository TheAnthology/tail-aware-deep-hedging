"""Heston market for the published Deep Hedging paper, Sections 5.1--5.2.

The variance skeleton is sampled from the exact CIR transition distribution,
including the paper's strongly non-Feller parameters (xi=2, CIR df=.04).
Stock increments use the conditional Broadie--Kaya decomposition, with the same
variance increment and an independent residual Gaussian. Integrated variance
is approximated by left endpoint sums by default. The paper specifies a
piecewise-constant approximation/sum but does not disclose its endpoint rule;
the left choice is therefore a declared local numerical convention. Trapezoid
and subdaily grids are exposed for diagnostics, never silently substituted.

Neither prices nor variances are clipped, resampled or martingale-corrected.
Exact CIR values may underflow to zero near the attainable boundary; those
zeros are retained and can be counted with simulation_diagnostics(). The
approximation's martingale/price bias must be measured separately.

Four float64 arrays of shape (1_000_000,31) occupy 992 MB (about .924 GiB),
before temporary arrays. Use path_chunks: an 8192-path chunk occupies about
8.13 MB for these arrays. Training/evaluation can consume chunks without
writing paths to disk.
"""
from __future__ import annotations

import math
from typing import Iterator

import numpy as np


PARAMETERS = dict(kappa=1.0, theta=0.04, rho=-0.7, xi=2.0,
                  v0=0.04, s0=100.0, steps=30, dt=1.0/365.0)


def _parameters(parameters=None):
    p = dict(PARAMETERS)
    if parameters is not None:
        p.update(parameters)
    if p['kappa'] <= 0 or p['theta'] <= 0 or p['xi'] <= 0:
        raise ValueError('Positive CIR parameters required')
    if not -1 <= p['rho'] <= 1 or p['v0'] < 0 or p['s0'] <= 0:
        raise ValueError('Invalid correlation or initial state')
    if int(p['steps']) != p['steps'] or p['steps'] < 1 or p['dt'] <= 0:
        raise ValueError('Positive integral daily step count and dt required')
    p['steps'] = int(p['steps'])
    return p


def L(t, variance, parameters=None):
    """Conditional expected future integrated variance, equation (31)."""
    p = _parameters(parameters)
    horizon = p['steps']*p['dt']
    remaining = horizon-np.asarray(t)
    if np.any(remaining < -1e-14):
        raise ValueError('t exceeds maturity')
    # expm1 avoids cancellation near maturity; this is an algebraic evaluation
    # of the same formula, not an alteration of model dynamics.
    factor = -np.expm1(-p['kappa']*remaining)/p['kappa']
    return (np.asarray(variance)-p['theta'])*factor+p['theta']*remaining


def L_v(t, parameters=None):
    """Variance derivative of L, denominator in the model swap hedge."""
    p = _parameters(parameters)
    remaining = p['steps']*p['dt']-np.asarray(t)
    if np.any(remaining < -1e-14):
        raise ValueError('t exceeds maturity')
    return -np.expm1(-p['kappa']*remaining)/p['kappa']


def cir_transition(variance, dt, rng, parameters=None):
    """Exact noncentral-chi-square transition; no Feller fix or truncation."""
    p = _parameters(parameters)
    variance = np.asarray(variance, dtype=np.float64)
    if dt <= 0 or np.any(variance < 0) or not np.isfinite(variance).all():
        raise ValueError('Finite nonnegative variance and positive dt required')
    decay = math.exp(-p['kappa']*dt)
    c = p['xi']**2*(-math.expm1(-p['kappa']*dt))/(4*p['kappa'])
    df = 4*p['kappa']*p['theta']/p['xi']**2
    noncentrality = decay*variance/c
    return c*rng.noncentral_chisquare(df, noncentrality)


def sample_paths(n, rng, quadrature='left', substeps=1, parameters=None):
    """Return daily spot, variance, swap and cumulative integrated variance.

    Variance endpoints are exact. Given an approximation J to the variance
    integral over a substep, the stock log increment is

      rho/xi*(dV-kappa*theta*dt) + (rho*kappa/xi-.5)*J
        + sqrt((1-rho**2)*J)*Z.

    This coupling preserves the variance-driver contribution. Drawing an exact
    CIR endpoint and a separate uncorrelated lognormal stock increment would
    not implement the paper's rho=-.7 market.
    """
    p = _parameters(parameters)
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError('n must be a positive integer')
    if not isinstance(substeps, (int, np.integer)) or substeps < 1:
        raise ValueError('substeps must be a positive integer')
    if quadrature not in ('left', 'trapezoid'):
        raise ValueError('quadrature must be left or trapezoid')
    shape = (int(n),p['steps']+1)
    spot = np.empty(shape,dtype=np.float64)
    variance = np.empty(shape,dtype=np.float64)
    swap = np.empty(shape,dtype=np.float64)
    integrated = np.empty(shape,dtype=np.float64)
    v = np.full(n,p['v0'],dtype=np.float64)
    log_s = np.full(n,math.log(p['s0']),dtype=np.float64)
    accumulated = np.zeros(n,dtype=np.float64)
    spot[:,0] = p['s0']
    variance[:,0] = v
    integrated[:,0] = 0.
    swap[:,0] = L(0.,v,p)
    h = p['dt']/substeps
    independent_scale = math.sqrt(1-p['rho']**2)
    for day in range(1,p['steps']+1):
        for _ in range(substeps):
            next_v = cir_transition(v,h,rng,p)
            J = v*h if quadrature == 'left' else .5*(v+next_v)*h
            independent_normal = rng.standard_normal(n)
            log_s += (p['rho']/p['xi']*(next_v-v-p['kappa']*p['theta']*h)
                      +(p['rho']*p['kappa']/p['xi']-.5)*J
                      +independent_scale*np.sqrt(J)*independent_normal)
            accumulated += J
            v = next_v
        spot[:,day] = np.exp(log_s)
        variance[:,day] = v
        integrated[:,day] = accumulated
        swap[:,day] = accumulated+L(day*p['dt'],v,p)
    arrays = dict(spot=spot,variance=variance,swap=swap,
                  integrated_variance=integrated)
    if any(not np.isfinite(x).all() for x in arrays.values()):
        raise RuntimeError('Nonfinite simulated state; no clipping permitted')
    return arrays


def path_chunks(total, chunk_size=8192, seed=1701, quadrature='left',
                substeps=1, parameters=None) -> Iterator[dict[str,np.ndarray]]:
    """Independent path chunks from one RNG stream; bounded retained arrays.

    The chunk size is part of reproducibility: changing it changes which RNG
    draws are assigned to a path, though the target sampling law is unchanged.
    """
    if not isinstance(total,(int,np.integer)) or total < 1:
        raise ValueError('total must be a positive integer')
    if not isinstance(chunk_size,(int,np.integer)) or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer')
    rng = np.random.default_rng(seed)
    generated = 0
    while generated < total:
        count = min(chunk_size,total-generated)
        yield sample_paths(count,rng,quadrature,substeps,parameters)
        generated += count


def sample_tensors(data, strike=100.0):
    """NN features, two-asset holding-period price changes, and call liability.

    Previous holdings are deliberately not inserted here: the NN supplies them
    recursively while retaining their gradient connection. Neural arrays are
    float32; evaluation uses the original float64 daily market arrays.
    """
    spot = np.asarray(data['spot'])
    variance = np.asarray(data['variance'])
    swap = np.asarray(data['swap'])
    if spot.ndim != 2 or variance.shape != spot.shape or swap.shape != spot.shape:
        raise ValueError('Expected equal (paths,dates) market arrays')
    features = np.stack([np.log(spot[:,:-1]),variance[:,:-1]],axis=-1)
    changes = np.stack([np.diff(spot,axis=1),np.diff(swap,axis=1)],axis=-1)
    payoff = np.maximum(spot[:,-1]-strike,0.)
    return features.astype('float32'),changes.astype('float32'),payoff.astype('float32')


def pnl(data, holdings, premium, strike=100.0):
    """Terminal P&L in currency units; both stock and variance swap gains."""
    spot = np.asarray(data['spot'],dtype=np.float64)
    swap = np.asarray(data['swap'],dtype=np.float64)
    holdings = np.asarray(holdings,dtype=np.float64)
    expected = (spot.shape[0],spot.shape[1]-1,2)
    if holdings.shape != expected:
        raise ValueError(f'Expected holdings shape {expected}, got {holdings.shape}')
    changes = np.stack([np.diff(spot,axis=1),np.diff(swap,axis=1)],axis=-1)
    return (premium-np.maximum(spot[:,-1]-strike,0.)
            +np.sum(holdings*changes,axis=(1,2)))


def empirical_cvar(loss, alpha):
    if not 0 <= alpha < 1:
        raise ValueError('alpha must lie in [0,1)')
    x = np.sort(np.asarray(loss,dtype=np.float64))[::-1]
    if x.ndim != 1 or not len(x):
        raise ValueError('Expected nonempty loss vector')
    mass = len(x)*(1-alpha)
    whole = int(np.floor(mass));fraction = mass-whole
    total = x[:whole].sum()+(fraction*x[whole] if fraction > 1e-10 else 0.)
    return float(total/mass)


def metrics(data, holdings, premium, strike=100.0):
    pl = pnl(data,holdings,premium,strike)
    loss = -pl
    h = np.asarray(holdings,dtype=np.float64)
    bounded = np.concatenate([np.zeros_like(h[:,:1,:]),h,np.zeros_like(h[:,:1,:])],axis=1)
    turnover = np.abs(np.diff(bounded,axis=1)).sum(axis=1)
    return dict(samples=len(pl),mean_pnl=float(pl.mean()),mse=float(np.mean(pl**2)),
                probability_loss=float(np.mean(pl < 0)),cvar50=empirical_cvar(loss,.5),
                cvar95=empirical_cvar(loss,.95),cvar99=empirical_cvar(loss,.99),
                mean_turnover_stock=float(turnover[:,0].mean()),
                mean_turnover_swap=float(turnover[:,1].mean()),
                max_absolute_stock_holding=float(np.abs(h[:,:,0]).max()),
                max_absolute_swap_holding=float(np.abs(h[:,:,1]).max()),
                minimum_pnl=float(pl.min()),maximum_pnl=float(pl.max()))


def simulation_diagnostics(data, strike=100.0, parameters=None):
    p = _parameters(parameters)
    spot = data['spot'];variance = data['variance'];swap = data['swap']
    n = len(spot)
    payoff = np.maximum(spot[:,-1]-strike,0.)
    return dict(paths=n,terminal_stock_mean=float(spot[:,-1].mean()),
                terminal_stock_mean_standard_error=float(spot[:,-1].std(ddof=1)/np.sqrt(n)) if n>1 else None,
                terminal_swap_mean=float(swap[:,-1].mean()),
                terminal_swap_mean_standard_error=float(swap[:,-1].std(ddof=1)/np.sqrt(n)) if n>1 else None,
                initial_swap=float(swap[0,0]),
                expected_terminal_variance=float(p['theta']+(p['v0']-p['theta'])*np.exp(-p['kappa']*p['steps']*p['dt'])),
                terminal_variance_mean=float(variance[:,-1].mean()),
                exact_zero_variance_states=int(np.count_nonzero(variance == 0)),
                total_variance_states=int(variance.size),
                call_payoff_mean=float(payoff.mean()),
                call_payoff_mean_standard_error=float(payoff.std(ddof=1)/np.sqrt(n)) if n>1 else None,
                minimum_spot=float(spot.min()),maximum_spot=float(spot.max()))


def self_checks(seed=812, n=20000):
    """Bounded implementation checks, not a claim of discretization accuracy."""
    p = _parameters();dt=p['dt'];rng=np.random.default_rng(seed)
    v = np.full(n,p['v0']);next_v=cir_transition(v,dt,rng,p)
    decay=np.exp(-p['kappa']*dt)
    expected_mean=p['theta']+(p['v0']-p['theta'])*decay
    expected_var=(p['v0']*p['xi']**2*decay*(1-decay)/p['kappa']
                  +p['theta']*p['xi']**2*(1-decay)**2/(2*p['kappa']))
    assert abs(next_v.mean()-expected_mean) < 6*np.sqrt(expected_var/n)
    # An independent scalar formula verifies conditional BK coupling and the
    # daily skeleton against the vector implementation at identical RNG draws.
    manual_rng=np.random.default_rng(seed+1);a=sample_paths(8,manual_rng)
    manual_rng=np.random.default_rng(seed+1)
    vv=np.full(8,p['v0']);ss=np.full(8,np.log(p['s0']));A=np.zeros(8)
    c=p['xi']**2*(1-np.exp(-p['kappa']*dt))/(4*p['kappa'])
    df=4*p['kappa']*p['theta']/p['xi']**2
    for k in range(1,31):
        # Use stable expm1 c, matching the exact transition algebra at float
        # precision while keeping the update formula independent of helpers.
        c=p['xi']**2*(-np.expm1(-p['kappa']*dt))/(4*p['kappa'])
        vn=c*manual_rng.noncentral_chisquare(df,np.exp(-p['kappa']*dt)*vv/c)
        z=manual_rng.standard_normal(8);J=vv*dt
        ss += p['rho']/p['xi']*(vn-vv-p['kappa']*p['theta']*dt)+(p['rho']*p['kappa']/p['xi']-.5)*J+np.sqrt((1-p['rho']**2)*J)*z
        A+=J;vv=vn
        assert np.allclose(a['spot'][:,k],np.exp(ss),rtol=1e-13,atol=1e-13)
        assert np.allclose(a['variance'][:,k],vv,rtol=1e-13,atol=1e-13)
        assert np.allclose(a['integrated_variance'][:,k],A,rtol=1e-13,atol=1e-13)
    assert np.allclose(a['swap'][:,-1],a['integrated_variance'][:,-1],rtol=0,atol=1e-15)
    assert np.allclose(a['swap'][:,0],p['theta']*30/365)
    features,changes,payoff=sample_tensors(a)
    assert features.shape == (8,30,2) and changes.shape == (8,30,2)
    assert np.all(features[:,0,0] == np.float32(np.log(100.)))
    assert np.all(features[:,0,1] == np.float32(.04))
    # Hold-one-unit identities telescope independently for each asset.
    for asset in (0,1):
        h=np.zeros((8,30,2));h[:,:,asset]=1.
        market=a['spot'] if asset==0 else a['swap']
        expected=7-payoff.astype('float64')+market[:,-1]-market[:,0]
        # payoff float32 conversion is deliberately excluded from exact P&L.
        expected=7-np.maximum(a['spot'][:,-1]-100,0)+market[:,-1]-market[:,0]
        assert np.allclose(pnl(a,h,7),expected,rtol=1e-12,atol=1e-12)
    assert np.isclose(empirical_cvar([1,2,3,4],.625),(4+.5*3)/1.5)
    return dict(status='PASS',cir_df=df,feller_left=2*p['kappa']*p['theta'],
                feller_right=p['xi']**2,cir_one_day_mean=float(next_v.mean()),
                cir_one_day_theoretical_mean=float(expected_mean),
                cir_one_day_variance=float(next_v.var()),cir_one_day_theoretical_variance=float(expected_var),
                checks=['Exact CIR conditional moments','Independent BK stock/variance daily recursion',
                        'Cumulative variance swap including maturity','Feature/price-change alignment',
                        'Two-asset self-financing P&L identities','Empirical CVaR'])


if __name__ == '__main__':
    import json
    print(json.dumps(self_checks(),indent=2))
