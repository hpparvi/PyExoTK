import numpy as np
import math

class PriorSet(object):
    def __init__(self, priors):
        self.priors = priors
        self.ndim  = len(priors)

        self.pmins = np.array([p.min() for p in self.priors])
        self.pmaxs = np.array([p.max() for p in self.priors])
        self.bounds= np.array([self.pmins,self.pmaxs]).T

        self.sbounds= np.array([[p.center-p.squeeze*0.5*p.width,
                                 p.center+p.squeeze*0.5*p.width] for p in self.priors])


    def generate_pv_population(self, npop):
        return np.array([[((p.random()[0]-p.center)*p.squeeze)+p.center for p in self.priors] for i in range(npop)])

    def c_log_prior(self, pv):
        return np.sum([p.log(v) for p,v in zip(self.priors, pv)])

    def c_prior(self, pv):
        return math.exp(self.c_log_prior(pv))


class Prior(object):
    def __init__(self, a, b, name='', description='', unit='', squeeze=1.):
        self.a = float(a)
        self.b = float(b)
        self.center= 0.5*(a+b)
        self.width = b - a
        self.squeeze = squeeze

        self.name = name
        self.description = description
        self.unit = unit

    def limits(self): return self.a, self.b 
    def min(self): return self.a
    def max(self): return self.b
   
 
class UniformPrior(Prior):
    def __init__(self, a, b, name='', description='', unit='', squeeze=1.):
        super(UniformPrior, self).__init__(a,b,name,description,unit,squeeze)
        self._f = 1. / self.width
        self._lf = math.log(self._f)

    def __call__(self, x, pv=None):
        if isinstance(x, np.ndarray):
            return np.where((self.a < x) & (x < self.b), self._f, 1e-80 * np.ones(x.size))
        else:
            return self._f if self.a < x < self.b else 1e-80

    def log(self, x, pv=None):
        if isinstance(x, np.ndarray):
            return np.where((self.a < x) & (x < self.b), self._lf, -1e18 * np.ones(x.size))
        else:
            return self._lf if self.a < x < self.b else -1e18

    def random(self, size=1):
        return np.random.uniform(self.a, self.b, size=size)


class JeffreysPrior(Prior):
    def __init__(self, a, b, name='', description='', unit='', squeeze=1.):
        super(JeffreysPrior, self).__init__(a,b,name,description,unit,squeeze)
        self._f = math.log(b/a)

    def __call__(self, x, pv=None):
        if isinstance(x, np.ndarray):
            return np.where((self.a < x) & (x < self.b), 1. / (x*self._f), 1e-80 * np.ones(x.size))
        else:
            return 1. / (x*self._f) if self.a < x < self.b else 1e-80

    def log(self, x, pv=None):
        if isinstance(x, np.ndarray):
            return np.where((self.a < x) & (x < self.b), math.log(1. / (x*self._f)), -1e18 * np.ones(x.size))
        else:
            return math.log(1. / (x*self._f)) if self.a < x < self.b else -1e18

    def random(self, size=1):
        return np.random.uniform(self.a, self.b, size=size)


class NormalPrior(Prior):
    def __init__(self, mean, std, name='', description='', unit='', lims=None, limsigma=5, squeeze=1.):
        lims = lims or (mean-limsigma*std, mean+limsigma*std)
        super(NormalPrior, self).__init__(*lims, name=name, description=description, unit=unit,squeeze=squeeze)
        self.mean = float(mean)
        self.std = float(std)
        self._f1 = 1./ math.sqrt(2.*math.pi*std*std)
        self._lf1 = math.log(self._f1)
        self._f2 = 1./ (2.*std*std)

    def __call__(self, x, pv=None):
        if isinstance(x, np.ndarray):
            return np.where((self.a < x) & (x < self.b),  self._f1 * np.exp(-(x-self.mean)**2 * self._f2), 1e-80 * np.ones(x.size))
        else:
            return self._f1 * exp(-(x-self.mean)**2 * self._f2) if self.a < x < self.b else 1e-80

    def log(self, x, pv=None):
        if isinstance(x, np.ndarray):
            return np.where((self.a < x) & (x < self.b),  self._lf1 - (x-self.mean)**2 * self._f2, -1e18 * np.ones(x.size))
        else:
            return self._lf1 -(x-self.mean)**2*self._f2 if self.a < x < self.b else -1e18

    def random(self, size=1):
        return np.random.uniform(self.a, self.b, size=size) #normal(self.mean, self.std, size)
    

UP = UniformPrior
JP = JeffreysPrior
GP = NormalPrior
NP = NormalPrior 
