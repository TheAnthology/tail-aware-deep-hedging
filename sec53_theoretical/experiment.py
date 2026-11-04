"""Heston adapter: project Section 5.3, originating from paper Section 5.2."""
import numpy as np
from . import market, experts, residual as engine
from .liability import HestonLiability


class Experiment:
    def __init__(self, window=0, cache_dir='runs/cache/heston'):
        self.liability = HestonLiability(cache_dir)
        self.q = self.liability.q
        self.metadata = {'market': market.PARAMETERS, 'quadrature': 'left',
                         'historical_data': False, 'original_paper_section': '5.2'}

    def sample(self, n, rng, with_liability=False):
        raw = market.sample_paths(n, rng)
        f, changes, payoff = market.sample_tensors(raw)
        data = {'f': f, 'changes': np.stack([np.diff(raw['spot']), np.diff(raw['swap'])], axis=-1),
                'payoff': np.maximum(raw['spot'][:,-1]-100., 0.), '_raw': raw}
        if with_liability:
            data['liability'] = self.liability.price(raw)
        return data

    def benchmark(self, data):
        return market.pnl(data['_raw'], self.liability.grid.hedge_paths(data['_raw']), self.q)

    def historical(self):
        return None

    expert = staticmethod(experts.CVaRExpert)
    expert_step = staticmethod(experts.make_step)
    calibrate = staticmethod(experts.calibrate)
    predict = staticmethod(experts.predict)
    residual = staticmethod(lambda scale, budget, q, seed: engine.ResidualCombiner(scale, budget, q, seed))
    residual_step = staticmethod(engine.make_step)
    forward = staticmethod(engine.make_forward)
    evaluate = staticmethod(engine.evaluate)
    tensors = staticmethod(engine.tensors)
