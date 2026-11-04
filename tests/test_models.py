import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import tensorflow as tf
from common.risk import empirical_cvar, tail95_tf
from common.io import save_weights, load_weights
from sec53_theoretical import market as hm
from sec53_theoretical.experts import CVaRExpert as HE, calibrate as hcal
from sec53_theoretical.residual import ResidualCombiner as HR
from sec55_sp500.models import CVaRExpert as SE, StockResidual as SR, calibrate as scal, market_data
from sec55_sp500.market import paths


class NumericalChecks(unittest.TestCase):
    def test_fractional_cvar_and_tensor(self):
        self.assertEqual(empirical_cvar([1,2,3,4],.75),4.)
        self.assertAlmostEqual(empirical_cvar([1,2,3,4],.6),3.625)
        x=np.random.default_rng(1).normal(size=123)
        self.assertAlmostEqual(float(tail95_tf(tf.constant(x))),empirical_cvar(x,.95),places=12)

    def test_markets_and_feature_causality(self):
        h=hm.sample_paths(128,np.random.default_rng(2))
        self.assertTrue(np.isfinite(h['spot']).all())
        self.assertTrue((h['variance']>=0).all())
        np.testing.assert_array_equal(h['spot'][:,0],100.)
        p=np.array([0.,.001,.05,.1,.85,6.])
        s,v=paths(p,{'h0':.02,'epsilon0':.01},100.,128,np.random.default_rng(3))
        a=market_data(s,v)
        s[:,16:]*=1.2;v[:,16:]*=1.5
        b=market_data(s,v)
        np.testing.assert_array_equal(a['f'][:,:4],b['f'][:,:4])
        np.testing.assert_array_equal(a['liability'][:,:4],b['liability'][:,:4])

    def test_bn_calibration_freezes_learned_parameters_and_reload(self):
        rng=np.random.default_rng(5)
        for cls,cal,steps in [(HE,hcal,30),(SE,scal,6)]:
            model=cls();model.initialize()
            f=np.stack([np.log(100)+rng.normal(0,.02,(64,steps)),rng.uniform(.001,.08,(64,steps))],-1).astype('float32')
            before=[x.numpy().copy() for x in model.trainable_variables]
            cal(model,f)
            for a,b in zip(before,model.trainable_variables):np.testing.assert_array_equal(a,b.numpy())
            a=model(f,training=False).numpy()
            b=np.concatenate([model(f[i:i+1],training=False).numpy() for i in range(len(f))])
            np.testing.assert_allclose(a,b,atol=1e-4,rtol=1e-5)
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'weights.npz';save_weights(model,path)
                replay=cls();replay.initialize();load_weights(replay,path)
                np.testing.assert_array_equal(a,replay(f,training=False).numpy())

    def test_residual_zero_head_accounting_and_causality(self):
        rng=np.random.default_rng(6)
        for steps,assets in [(30,2),(6,1)]:
            shape=(32,steps,2) if assets==2 else (32,steps)
            f=np.stack([np.log(100)+rng.normal(0,.02,(32,steps)),rng.uniform(.001,.08,(32,steps))],-1).astype('float32')
            ch=rng.normal(0,.1,shape);hc=rng.normal(0,.3,shape);hp=rng.normal(0,.5,shape)
            liability=rng.uniform(0,2,(32,steps));z=rng.uniform(0,4,32)
            inputs=[tf.constant(a) for a in [f,ch,hc,hp,liability,z]]
            m=HR([.5,100.],2.,1.7) if assets==2 else SR(.5,2.,1.7)
            pl,h,c,s=m(tuple(inputs))
            np.testing.assert_array_equal(h.numpy(),hc)
            np.testing.assert_array_equal(c.numpy(),np.zeros(shape))
            np.testing.assert_allclose(pl.numpy(),1.7+(hc*ch).sum(tuple(range(1,hc.ndim)))-z,atol=1e-12)
            # Activate the correction so the causal test isn't vacuous.
            m.out.kernel.assign(tf.ones_like(m.out.kernel)*.03)
            before=m(tuple(inputs))[1].numpy()
            altered=[a.numpy().copy() for a in inputs]
            altered[1][:,2:]+=1.
            altered[0][:,3:]+=1.
            altered[4][:,3:]+=2.
            after=m(tuple(tf.constant(a) for a in altered))[1].numpy()
            np.testing.assert_array_equal(before[:,:3],after[:,:3])
            self.assertGreater(float(np.max(np.abs(before[:,3:]-after[:,3:]))),0.)

    def test_released_weight_files(self):
        root=Path(__file__).resolve().parents[1]
        record=json.loads((root/'tests/recorded_checkpoint_parity.json').read_text())
        self.assertTrue(record['all_exact'])
        self.assertEqual(len(record['checks']),12)
        for row in record['checks']:
            self.assertEqual(hashlib.sha256((root/row['file']).read_bytes()).hexdigest(),row['sha256'])


if __name__=='__main__': unittest.main()
