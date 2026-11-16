"""Past-only GJR-GARCH calibration and real underlying-path evaluation."""

import numpy as np
from . import market, models, training
from .paper_hedge import PaperDeepHedge, paper_features


class Experiment:
    paper_model = staticmethod(PaperDeepHedge)

    def __init__(self, window=0, cache_dir=None):
        row, self.history, self.returns = market.window_data(window)
        self.params, self.state, fit = market.fit_garch(self.returns[:1187])
        self.q = market.bs_premium(100.0, fit["initial_sigma0_daily"])
        self.metadata = {
            "window": row,
            "fit": fit,
            "original_spot": float(self.history[1187]),
            "historical_data": True,
            "units": "S0=K=100",
        }

    def sample(self, n, rng, with_liability=False):
        prices, vol = market.paths(self.params, self.state, 100.0, n, rng)
        data = models.market_data(prices, vol)
        data["_raw"] = (prices, vol)
        return data

    def paper_holdings(self, model, data):
        # The paper network was trained on absolute index levels. Holdings are shares.
        prices, _ = data["_raw"]
        return models.predict(model, paper_features(prices, self.metadata["original_spot"]))

    def benchmark(self, data):
        prices, vol = data["_raw"]
        return market.pnl(
            prices,
            market.bs_hedge(prices, vol, 100.0, models.HEDGE_DAYS),
            self.q,
            100.0,
            models.HEDGE_DAYS,
        )

    def historical(self):
        prices, vol = market.observed_path(
            self.params, self.state, 100.0, self.returns[1187:]
        )
        np.testing.assert_allclose(
            prices[0], 100.0 * self.history[1187:] / self.history[1187], atol=1e-10
        )
        data = models.market_data(prices, vol)
        data["_raw"] = (prices, vol)
        return data

    expert = staticmethod(models.CVaRExpert)
    expert_step = staticmethod(training.make_step)
    calibrate = staticmethod(models.calibrate)
    predict = staticmethod(models.predict)
    novel_model_1 = staticmethod(
        lambda scale, budget, q, seed: models.Novel_model_1(scale, budget, q, seed)
    )
    policy_step = staticmethod(models.make_policy_step)
    forward = staticmethod(models.make_policy_forward)
    evaluate = staticmethod(models.evaluate_policy)
    tensors = staticmethod(models.policy_tensors)
