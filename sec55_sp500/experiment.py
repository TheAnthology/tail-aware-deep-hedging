"""Past-only GJR-GARCH calibration and real underlying-path evaluation."""
import numpy as np
from . import market, models, training


class Experiment:
    def __init__(self, window=0, cache_dir=None):
        row, self.history, self.returns = market.window_data(window)
        self.params, self.state, fit = market.fit_garch(self.returns[:1187])
        self.q = market.bs_premium(100., fit['initial_sigma0_daily'])
        self.metadata = {'window': row, 'fit': fit, 'original_spot': float(self.history[1187]),
                         'historical_data': True, 'units': 'S0=K=100'}

    def sample(self, n, rng, with_liability=False):
        prices, vol = market.paths(self.params, self.state, 100., n, rng)
        data = models.market_data(prices, vol)
        data['_raw'] = (prices, vol)
        return data

    def benchmark(self, data):
        prices, vol = data['_raw']
        return market.pnl(prices, market.bs_hedge(prices, vol, 100., models.HEDGE_DAYS), self.q, 100., models.HEDGE_DAYS)

    def historical(self):
        prices, vol = market.observed_path(self.params, self.state, 100., self.returns[1187:])
        np.testing.assert_allclose(prices[0], 100.*self.history[1187:]/self.history[1187], atol=1e-10)
        data = models.market_data(prices, vol)
        data['_raw'] = (prices, vol)
        return data

    expert = staticmethod(models.CVaRExpert)
    expert_step = staticmethod(training.make_step)
    calibrate = staticmethod(models.calibrate)
    predict = staticmethod(models.predict)
    residual = staticmethod(lambda scale, budget, q, seed: models.StockResidual(scale, budget, q, seed))
    residual_step = staticmethod(models.make_res_step)
    forward = staticmethod(models.make_res_forward)
    evaluate = staticmethod(models.evaluate_res)
    tensors = staticmethod(models.residual_tensors)
