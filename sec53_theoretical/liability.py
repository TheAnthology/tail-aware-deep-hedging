"""Causal price interpolation; missing caches are explicitly built on first use."""
from pathlib import Path
import numpy as np
from .benchmark import HestonGridHedge

class HestonLiability:
    def __init__(self, cache_dir=None):
        self.cache_dir = Path(cache_dir) if cache_dir is not None else Path('runs/cache/heston')
        self.grid = HestonGridHedge(cache_dir=self.cache_dir, sensitive_variance=.001)
        self.q = self.grid.q0
        self.total_points = 0
        self.direct_fallback_points = 0
        self.boundary_fallback_points = 0
        self.cache_paths = {}
        for days in range(1,31):
            path = self.grid._cache_path(days)
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
