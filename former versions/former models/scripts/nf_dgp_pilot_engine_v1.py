"""Small scalar spline density, independent of frozen hedge training.
Rational-quadratic spline equations: Durkan et al. (2019), arXiv:1906.04032.
No nflows code or dependency is imported. CPU float64 throughout.
"""
import math
import numpy as np
import torch
from scipy import stats, integrate
from torch import nn

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

def tensor(x):return torch.as_tensor(x,dtype=torch.float64)

class ScalarSpline(nn.Module):
    def __init__(self,nu,seed=123):
        super().__init__();g=torch.Generator().manual_seed(seed);self.nu=float(nu)
        self.width=nn.Parameter(.01*torch.randn(16,generator=g))
        self.height=nn.Parameter(.01*torch.randn(16,generator=g))
        self.derivative=nn.Parameter(torch.full((15,),math.log(math.expm1(.999)))+.01*torch.randn(15,generator=g))
        self.base=torch.distributions.StudentT(self.nu,scale=math.sqrt((self.nu-2)/self.nu))
        nodes,weights=np.polynomial.legendre.leggauss(16)
        self.register_buffer('nodes',tensor(nodes));self.register_buffer('weights',tensor(weights))
    def geometry(self):
        w=8*(.001+.984*torch.softmax(self.width,0));h=8*(.001+.984*torch.softmax(self.height,0))
        x=torch.cat([tensor([-4.]),-4+torch.cumsum(w,0)]);y=torch.cat([tensor([-4.]),-4+torch.cumsum(h,0)])
        d=torch.cat([tensor([1.]),.001+torch.nn.functional.softplus(self.derivative),tensor([1.])])
        return x,y,w,h,d
    def transform(self,value,inverse=False):
        value=tensor(value);x,y,w,h,d=self.geometry();inside=(value>=-4)&(value<=4)
        q=value.clamp(-4,4);bins=torch.searchsorted((y if inverse else x).detach(),q.detach(),right=True).sub(1).clamp(0,15)
        x0=x[bins];y0=y[bins];ww=w[bins];hh=h[bins];s=hh/ww;dl=d[bins];dr=d[bins+1];bend=dl+dr-2*s
        if inverse:
            v=q-y0;a=v*bend+hh*(s-dl);b=hh*dl-v*bend;c=-s*v
            disc=b*b-4*a*c
            if torch.any(disc < -1e-10):raise FloatingPointError('Invalid spline discriminant')
            theta=2*c/(-b-torch.sqrt(disc.clamp_min(0)))
        else:theta=(q-x0)/ww
        mix=theta*(1-theta);den=s+bend*mix
        derivative=s*s*(dr*theta*theta+2*s*mix+dl*(1-theta)**2)/(den*den)
        out=x0+ww*theta if inverse else y0+hh*(s*theta*theta+dl*mix)/den
        ld=torch.log(derivative)*(-1 if inverse else 1)
        return torch.where(inside,out,value),torch.where(inside,ld,torch.zeros_like(value))
    def moments(self):
        x,_,w,_,_=self.geometry()
        u=x[:-1,None]+w[:,None]*(self.nodes+1)/2
        weights=w[:,None]*self.weights/2*torch.exp(self.base.log_prob(u))
        f,_=self.transform(u.reshape(-1));f=f.reshape(u.shape)
        m=torch.sum(weights*(f-u));second=1+torch.sum(weights*(f*f-u*u))
        return m,second-m*m
    def log_prob(self,z):
        m,v=self.moments();y=tensor(z)*torch.sqrt(v)+m;u,ild=self.transform(y,True)
        return self.base.log_prob(u)+ild+.5*torch.log(v)
    def quantile(self,p):
        with torch.no_grad():
            m,v=self.moments();u=stats.t.ppf(p,self.nu)*math.sqrt((self.nu-2)/self.nu)
            f,_=self.transform(u);return ((f-m)/torch.sqrt(v)).numpy()
    def cdf(self,z):
        with torch.no_grad():
            m,v=self.moments();u,_=self.transform(tensor(z)*torch.sqrt(v)+m,True)
            return stats.t.cdf(u.numpy()/math.sqrt((self.nu-2)/self.nu),self.nu)
    def freeze_metadata(self,p):
        with torch.no_grad():
            m,v=self.moments();m=float(m);v=float(v);knots=self.geometry()[0].numpy()
            cut=float(self.transform(tensor(m),True)[0])
        scale=math.sqrt((self.nu-2)/self.nu)
        def f(u):
            with torch.no_grad():return float(self.transform(tensor(u))[0])
        pdf=lambda u:stats.t.pdf(u/scale,self.nu)/scale
        # Independent adaptive integration; identity tails handled analytically by base moments.
        mi=integrate.quad(lambda u:(f(u)-u)*pdf(u),-4,4,points=knots[1:-1],epsabs=1e-10)[0]
        sec=1+integrate.quad(lambda u:(f(u)**2-u*u)*pdf(u),-4,4,points=knots[1:-1],epsabs=1e-10)[0]
        ki=integrate.quad(lambda u:((f(u)-m)**2/v)*pdf(u),-np.inf,min(cut,-4),epsabs=1e-9)[0]
        if cut>-4:
            ki+=integrate.quad(lambda u:((f(u)-m)**2/v)*pdf(u),-4,cut,points=knots[(knots>-4)&(knots<cut)],epsabs=1e-9)[0]
        error=max(abs(mi-m),abs(sec-mi*mi-v));assert error<1e-7
        rho=float(p[2]+p[4]+p[3]*ki)
        if rho>=1:raise RuntimeError(f'Flow finite-second-moment condition fails: rho={rho}')
        return {'raw_mean':m,'raw_variance':v,'kappa':ki,'rho':rho,'moment_integration_max_error':error,'parameter_count':sum(v.numel() for v in self.parameters())}

class Law:
    def __init__(self,kind,nu,residuals=None,model=None):
        self.kind=kind;self.nu=float(nu);self.model=model;self.scale=math.sqrt((nu-2)/nu)
        if residuals is not None:self.bank=np.sort((residuals-np.mean(residuals))/np.std(residuals))
    def quantile(self,p):
        if self.kind=='flow':return self.model.quantile(p)
        if self.kind=='bootstrap':return self.bank[np.minimum((np.asarray(p)*len(self.bank)).astype(int),len(self.bank)-1)]
        if self.kind=='normal':return stats.norm.ppf(p)
        return stats.t.ppf(p,self.nu)*self.scale
    def cdf(self,z):
        if self.kind=='flow':return self.model.cdf(z)
        if self.kind=='bootstrap':return np.searchsorted(self.bank,z,side='right')/len(self.bank)
        if self.kind=='normal':return stats.norm.cdf(z)
        return stats.t.cdf(np.asarray(z)/self.scale,self.nu)

def state_sequence(p,state,returns):
    mu,omega,a,g,b,nu=p;h=state['h0'];eps=state['epsilon0'];seq=[]
    for r in returns:
        h=omega+(a+g*(eps<0))*eps*eps+b*h;seq.append(h);eps=100*r-mu
    return np.array(seq)

def simulate(p,state,z):
    mu,omega,a,g,b,nu=p;h=np.full(len(z),state['h0']);eps=np.full(len(z),state['epsilon0'])
    r=np.empty_like(z);hh=np.empty_like(z)
    for k in range(z.shape[1]):
        h=omega+(a+g*(eps<0))*eps*eps+b*h;eps=np.sqrt(h)*z[:,k];r[:,k]=(mu+eps)/100;hh[:,k]=h/10000
    assert np.isfinite(r).all() and np.isfinite(hh).all() and (hh>0).all()
    return r,hh

def crps(samples,y):
    x=np.sort(samples,axis=0);n=len(x);coef=(2*np.arange(1,n+1)-n-1).reshape((n,)+(1,)*(x.ndim-1))
    return np.mean(abs(x-y),axis=0)-np.sum(coef*x,axis=0)/(n*n)

def pinball(y,q,alpha):
    e=np.asarray(y)-q;return np.mean(np.maximum(alpha*e,(alpha-1)*e))

def path_scores(r,independent,y):
    n=len(r);energy=np.mean(np.linalg.norm(r-y,axis=1))-.5*np.mean(np.linalg.norm(r-independent,axis=1))
    vs=0.
    for lag in range(1,30):
        target=abs(y[lag:]-y[:-lag])**.5
        pred=np.mean(abs(r[:,lag:]-r[:,:-lag])**.5,axis=0)
        vs+=np.sum((pred-target)**2)/lag
    terminal=r.sum(1);actual=y.sum()
    return {'crps30':float(crps(terminal,actual)),'energy_score':float(energy),'variogram_score':float(vs),
        'terminal_actual_log_return':float(actual),'terminal_predictive_mean':float(terminal.mean()),
        'terminal_predictive_sd':float(terminal.std()),
        'terminal_quantiles':{str(a):float(np.quantile(terminal,a)) for a in [.01,.05,.5,.95,.99]},
        'pinball30':{str(a):float(pinball(actual,np.quantile(terminal,a),a)) for a in [.01,.05]},
        'mean_realized_variance':float(np.mean(np.sum(r*r,axis=1))),
        'mean_max_drawdown':float(np.mean(np.max(1-np.exp(np.cumsum(r,axis=1)-np.maximum.accumulate(np.column_stack([np.zeros(n),np.cumsum(r,axis=1)]),axis=1)[:,1:]),axis=1)))}

def one_day_scores(law,p,state,observed,n=20000):
    h=state_sequence(p,state,observed);sigma=np.sqrt(h)/100;mu=p[0]/100
    # Same innovation grid, exact distribution quantiles; no batch normalization.
    probs=(np.arange(n)+.5)/n;z=law.quantile(probs)
    samples=mu+z[:,None]*sigma;pit=law.cdf((observed-mu)/sigma)
    qs={a:mu+law.quantile(np.array(a))*sigma for a in [.01,.05]}
    return {'crps1_mean':float(np.mean(crps(samples,observed))),
        'crps1_each':crps(samples,observed).tolist(),'pit':np.asarray(pit).tolist(),
        'coverage':{str(a):{'exceedances':int(np.sum(observed<q)),'observations':len(observed),'pinball':float(pinball(observed,q,a)),'quantiles':q.tolist()} for a,q in qs.items()}}

def checks():
    m=ScalarSpline(6,123);x=torch.linspace(-7,7,1001);y,ld=m.transform(x);back,ild=m.transform(y,True)
    assert torch.max(abs(back-x))<1e-10 and torch.max(abs(ld+ild))<1e-10
    q=torch.linspace(-3.99,3.99,100,requires_grad=True);v,lj=m.transform(q)
    grad=torch.autograd.grad(v.sum(),q)[0];assert torch.max(abs(torch.log(grad)-lj))<1e-10
    loss=-m.log_prob(tensor([-2.,-.4,.3,1.6])).mean();loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())
    grid=np.linspace(-20,20,401);np.testing.assert_allclose(m.quantile(m.cdf(grid)),grid,atol=1e-7)
    a=np.array([1.,2.,4.]);assert abs(float(crps(a,2.))-(np.mean(abs(a-2))-.5*np.mean(abs(a[:,None]-a))))<1e-12
    return {'spline_inverse_logjac_autograd_cdf_quantile':True,'finite_parameter_gradients':True,'crps_matches_pairwise':True}
