"""Bounded CPU pilot under protocol_v1.json. No upstream code is executed."""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/deephedge_matplotlib')
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln, ndtr
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def variance_filter(x, p):
    mu, omega, alpha, gamma, beta, nu = p
    eps = x - mu
    h0 = float(np.var(x))
    forcing = omega + (alpha + gamma * (eps[:-1] < 0)) * eps[:-1]**2
    rest, _ = lfilter([1.0], [1.0, -beta], forcing, zi=[beta * h0])
    return eps, np.r_[h0, rest]


def fit_market(returns):
    x = returns * 100
    var = float(np.var(x))

    def nll(p):
        eps, h = variance_filter(x, p)
        if np.any(h <= 0) or not np.all(np.isfinite(h)):
            return 1e30
        nu = p[-1]
        log_density = (gammaln((nu+1)/2) - gammaln(nu/2)
                       - 0.5*np.log((nu-2)*np.pi*h)
                       - (nu+1)/2*np.log1p(eps**2 / ((nu-2)*h)))
        return float(-log_density.sum())

    fits = []
    for alpha, gamma, beta, nu in [(0.05, .1, .85, 8.), (.05, .15, .8, 5.), (.1, .05, .8, 15.)]:
        start = [np.mean(x), var*(1-alpha-gamma/2-beta), alpha, gamma, beta, nu]
        result = minimize(nll, start, method='SLSQP',
                          bounds=[(-5*np.sqrt(var), 5*np.sqrt(var)), (1e-9, 100*var),
                                  (0, .999), (0, 1.998), (0, .999), (2.05, 100)],
                          constraints=[{'type': 'ineq', 'fun': lambda p: .999-p[2]-p[3]/2-p[4]}],
                          options={'maxiter': 1000, 'ftol': 1e-9})
        fits.append(dict(success=bool(result.success), message=str(result.message),
                         nll=float(result.fun), iterations=int(result.nit), parameters=result.x.tolist()))
    valid = [r for r in fits if r['success'] and r['parameters'][2]+r['parameters'][3]/2+r['parameters'][4] <= .9990001]
    if not valid:
        raise RuntimeError(f'No converged GARCH fit: {fits}')
    chosen = min(valid, key=lambda r: r['nll'])
    p = np.array(chosen['parameters'])
    eps, h = variance_filter(x, p)
    h_next = p[1]+(p[2]+p[3]*(eps[-1]<0))*eps[-1]**2+p[4]*h[-1]
    return p, h_next, dict(parameter_names=['mu_percent', 'omega_percent_squared', 'alpha', 'gamma', 'beta', 'nu'],
                           parameters=p.tolist(), fits=fits, persistence=float(p[2]+p[3]/2+p[4]),
                           initial_forecast_variance_percent_squared=float(h_next),
                           initial_daily_volatility=float(np.sqrt(h_next)/100),
                           calibration_observations=len(x))


def simulate(p, h_next, s0, n, seed, days=30):
    rng = np.random.default_rng(seed)
    mu, omega, alpha, gamma, beta, nu = p
    prices = np.empty((n, days+1), dtype=np.float64)
    vols = np.empty((n, days+1), dtype=np.float64)
    prices[:, 0] = s0
    h = np.full(n, h_next)
    vols[:, 0] = np.sqrt(h)/100
    for k in range(days):
        z = rng.standard_t(nu, n)*np.sqrt((nu-2)/nu)
        eps = np.sqrt(h)*z
        prices[:, k+1] = prices[:, k]*np.exp((mu+eps)/100)
        h = omega+(alpha+gamma*(eps<0))*eps**2+beta*h
        vols[:, k+1] = np.sqrt(h)/100
    if not np.isfinite(prices).all():
        raise RuntimeError('Nonfinite simulated prices; no clipping is allowed')
    return prices, vols


def historical_path(p, h_next, s0, returns):
    mu, omega, alpha, gamma, beta, nu = p
    prices = np.r_[s0, s0*np.exp(np.cumsum(returns))][None, :]
    vols = np.empty_like(prices)
    h = h_next
    vols[0, 0] = np.sqrt(h)/100
    for k, r in enumerate(returns):
        eps = 100*r-mu
        h = omega+(alpha+gamma*(eps<0))*eps**2+beta*h
        vols[0, k+1] = np.sqrt(h)/100
    return prices, vols


def bs_price(s0, sigma_daily, horizon):
    v = sigma_daily*np.sqrt(horizon)
    return float(s0*(2*ndtr(v/2)-1))


def bs_positions(prices, vols, strike, days):
    s = prices[:, days]
    v = vols[:, days]*np.sqrt(30-np.array(days))
    return ndtr((np.log(s/strike)+.5*v*v)/v)


def empirical_cvar(loss, alpha):
    """Exact empirical ES, including fractional order-statistic weight."""
    x = np.sort(np.asarray(loss, dtype=np.float64))[::-1]
    mass = len(x)*(1-alpha)
    full = int(np.floor(mass))
    fraction = mass-full
    total = x[:full].sum() + (fraction*x[full] if fraction > 1e-10 else 0.)
    return float(total/mass)


def pnl_from_positions(prices, holdings, premium, strike, days):
    hedge_prices = prices[:, list(days)+[30]]
    payoff = np.maximum(prices[:, -1]-strike, 0)
    return premium-payoff+(holdings*np.diff(hedge_prices, axis=1)).sum(axis=1)


def metrics(prices, holdings, premium, strike, days):
    pnl = pnl_from_positions(prices, holdings, premium, strike, days)
    loss = -pnl
    turnover = np.abs(np.diff(np.c_[np.zeros(len(holdings)), holdings, np.zeros(len(holdings))], axis=1)).sum(axis=1)
    return dict(mean_pnl=float(pnl.mean()), mse=float(np.mean(pnl**2)),
                probability_loss=float(np.mean(loss>0)), probability_loss_above_premium=float(np.mean(loss>premium)),
                cvar75=empirical_cvar(loss, .75), cvar95=empirical_cvar(loss, .95),
                mean_turnover=float(turnover.mean()), mean_transaction_cost=0.,
                max_absolute_holdings=float(np.abs(holdings).max()), samples=len(pnl)), pnl


class Hedge(nn.Module):
    def __init__(self, steps):
        super().__init__()
        self.nets = nn.ModuleList([nn.Sequential(nn.Linear(2,20), nn.BatchNorm1d(20), nn.ReLU(),
                                                 nn.Linear(20,20), nn.BatchNorm1d(20), nn.ReLU(),
                                                 nn.Linear(20,1)) for _ in range(steps)])
        self.eta = nn.Parameter(torch.tensor(0., dtype=torch.float32))

    def forward(self, log_spots):
        previous = torch.zeros_like(log_spots[:, :1])
        positions = []
        for k, net in enumerate(self.nets):
            previous = net(torch.cat([log_spots[:, k:k+1], previous], dim=1))
            positions.append(previous)
        return torch.cat(positions, dim=1)


def tensors(prices, strike, premium, days):
    hp = prices[:, list(days)+[30]]
    return (torch.tensor(np.log(hp[:, :-1]), dtype=torch.float32),
            torch.tensor(np.diff(hp, axis=1)/premium, dtype=torch.float32),
            torch.tensor(np.maximum(prices[:, -1]-strike, 0)/premium, dtype=torch.float32))


def evaluate_network(model, prices, days, chunk=4096):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.tensor(np.log(prices[i:i+chunk, days]), dtype=torch.float32)).numpy()
                               for i in range(0, len(prices), chunk)])


def self_checks():
    p = np.array([0., .01, .05, .1, .8, 8.])
    x = np.array([.1, -.2, .3, -.1])
    eps, h = variance_filter(x, p)
    for k in range(1, len(x)):
        assert np.isclose(h[k], p[1]+(p[2]+p[3]*(eps[k-1]<0))*eps[k-1]**2+p[4]*h[k-1])
    # Prefix causality: appending future returns cannot change earlier volatility.
    assert np.allclose(historical_path(p, .1, 100, x/100)[1][:, :3],
                       historical_path(p, .1, 100, x[:2]/100)[1])
    prices = np.array([[100., 110., 120.]])
    assert empirical_cvar([1,2,3,4], .75) == 4.
    assert np.isclose(empirical_cvar([1,2,3,4], .625), (4+.5*3)/1.5)
    # Self-financing hold-one-share identity across the six trading intervals.
    grid = np.arange(31, dtype=float)[None, :]+100
    assert np.allclose(pnl_from_positions(grid, np.ones((1,6)), 7, 100, [0,5,10,15,20,25]), 7)
    torch.manual_seed(1)
    model = Hedge(6)
    model.eval()
    logs = torch.randn(12,6)
    changed = logs.clone(); changed[:,3:] += 5
    assert torch.allclose(model(logs)[:,:3], model(changed)[:,:3])
    # Verify the recurrent state influences a later position, and gradients propagate.
    a = model.nets[1](torch.cat([logs[:,1:2], torch.zeros(12,1)], 1))
    b = model.nets[1](torch.cat([logs[:,1:2], torch.ones(12,1)], 1))
    assert not torch.allclose(a,b)
    model(logs).sum().backward()
    assert model.nets[0][0].weight.grad.abs().sum() > 0
    print('Checks passed: GARCH recursion, causal filters/policy, CVaR, P&L identity, recurrence and gradients.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    self_checks()
    if args.check_only:
        return
    started = time.monotonic()
    protocol_path = ROOT/'protocol_v1.json'
    cfg = json.loads(protocol_path.read_text())
    out = ROOT/'results'/cfg['id']
    out.mkdir(parents=True, exist_ok=False)
    (out/'protocol.json').write_text(protocol_path.read_text())
    input_path = ROOT/'data/processed/paper_main_returns_1217.csv'
    with input_path.open() as f:
        rows = list(csv.DictReader(f))
    train_rows = [r for r in rows if r['split_30d']=='calibration']
    history_rows = [r for r in rows if r['split_30d']=='historical_test']
    assert len(train_rows)==cfg['calibration_returns'] and len(history_rows)==30
    calibration = np.array([float(r['log_return']) for r in train_rows])
    p, h_next, fit = fit_market(calibration)
    (out/'garch_fit.json').write_text(json.dumps(fit, indent=2)+'\n')
    print('GARCH fit:', json.dumps(fit), flush=True)
    s0 = strike = float(train_rows[-1]['adjusted_close'])
    premium = bs_price(s0, np.sqrt(h_next)/100, 30)
    days = cfg['rebalance_days']
    train_prices, _ = simulate(p, h_next, s0, cfg['n_train'], cfg['market_seed_train'])
    val_prices, _ = simulate(p, h_next, s0, cfg['n_validation'], cfg['market_seed_validation'])
    training = tensors(train_prices, strike, premium, days)
    torch.manual_seed(cfg['network_seed'])
    model = Hedge(len(days))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    shuffle_rng = np.random.default_rng(cfg['shuffle_seed'])
    initial_holdings = evaluate_network(model, val_prices, days)
    initial_metrics, _ = metrics(val_prices, initial_holdings, premium, strike, days)
    best_score = initial_metrics['cvar75']
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    log = [dict(epoch=0, validation_cvar75=best_score, mean_training_objective_scaled=None)]
    for epoch in range(1, cfg['epochs']+1):
        model.train()
        objectives = []
        permutation = shuffle_rng.permutation(cfg['n_train'])
        for lo in range(0, len(permutation), cfg['batch_size']):
            ix = permutation[lo:lo+cfg['batch_size']]
            log_spots, changes, payoff = [t[ix] for t in training]
            holdings = model(log_spots)
            loss = payoff-1-(holdings*changes).sum(1)
            objective = model.eta + torch.relu(loss-model.eta).mean()/(1-cfg['alpha'])
            if not torch.isfinite(objective):
                raise RuntimeError('Nonfinite objective')
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise RuntimeError('Nonfinite gradient')
            optimizer.step()
            objectives.append(float(objective.detach()))
        holdings = evaluate_network(model, val_prices, days)
        validation, _ = metrics(val_prices, holdings, premium, strike, days)
        score = validation['cvar75']
        log.append(dict(epoch=epoch, validation_cvar75=score,
                        mean_training_objective_scaled=float(np.mean(objectives))))
        if score < best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        print(f'Epoch {epoch:02d}/{cfg["epochs"]}: validation CVaR75={score:.4f}, best={best_score:.4f}', flush=True)
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), out/'selected_model.pt')
    with (out/'training_history.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(log[0])); writer.writeheader(); writer.writerows(log)
    # Test sets first generated/evaluated after checkpoint selection is complete.
    test_prices, test_vols = simulate(p, h_next, s0, cfg['n_test'], cfg['market_seed_test'])
    observed_returns = np.array([float(r['log_return']) for r in history_rows])
    historical_prices, historical_vols = historical_path(p, h_next, s0, observed_returns)
    assert np.allclose(historical_prices[0,1:], [float(r['adjusted_close']) for r in history_rows])
    results = dict(protocol_id=cfg['id'], protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                   input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
                   versions=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__),
                   initial_spot=s0, strike=strike, premium=premium, selected_epoch=best_epoch,
                   initial_validation_cvar75=initial_metrics['cvar75'], selected_validation_cvar75=best_score,
                   calibration_last_date=train_rows[-1]['date'], historical_test_first=history_rows[0]['date'],
                   historical_test_last=history_rows[-1]['date'], simulated={}, historical={})
    pnls = {}
    for split, prices, vols in [('simulated', test_prices, test_vols), ('historical', historical_prices, historical_vols)]:
        positions = dict(deep_cvar75=evaluate_network(model, prices, days),
                         black_scholes=bs_positions(prices, vols, strike, days),
                         unhedged=np.zeros((len(prices),len(days))))
        for name, holdings in positions.items():
            summary, pnl = metrics(prices, holdings, premium, strike, days)
            results[split][name] = summary
            if split=='simulated': pnls[name] = pnl
            else:
                results[split][name]['holdings'] = holdings[0].tolist()
                results[split][name]['terminal_pnl'] = float(pnl[0])
        if split=='historical':
            results['historical']['warning']='One realized path, not an estimate of tail probabilities or CVaR. Repeated single-path summary metrics are not distributional evidence.'
    # Paired test uncertainty for two pre-specified differences, conditional on simulator and trained model.
    difference = pnls['deep_cvar75']-pnls['black_scholes']
    probability_difference = (pnls['deep_cvar75']<0).astype(float)-(pnls['black_scholes']<0).astype(float)
    results['paired_test_mc_uncertainty']={}
    for name, d in [('mean_pnl_difference',difference),('probability_loss_difference',probability_difference)]:
        mean = float(d.mean()); se = float(d.std(ddof=1)/np.sqrt(len(d)))
        results['paired_test_mc_uncertainty'][name] = dict(estimate=mean, standard_error=se, normal95=[mean-1.96*se, mean+1.96*se])
    results['elapsed_seconds'] = time.monotonic()-started
    (out/'metrics.json').write_text(json.dumps(results, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    axes[0].plot([r['epoch'] for r in log], [r['validation_cvar75'] for r in log])
    axes[0].set(xlabel='Epoch', ylabel='Validation CVaR75 (index points)', title='CVaR baseline training')
    low, high = np.quantile(np.r_[pnls['deep_cvar75'],pnls['black_scholes']], [.002,.998])
    bins = np.linspace(low, high, 90)
    for name in ['deep_cvar75','black_scholes']:
        axes[1].hist(pnls[name], bins=bins, alpha=.5, density=True, label=name)
    axes[1].set(xlabel='Terminal P&L (index points)', ylabel='Density', title='Independent simulated test (central 99.6%)')
    axes[1].legend()
    fig.tight_layout(); fig.savefig(out/'pilot.png', dpi=150); plt.close(fig)
    print('RESULT', json.dumps(results), flush=True)


if __name__=='__main__':
    main()
