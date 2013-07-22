import math  as m
import numpy as np

from pytransit.gimenez import Gimenez 
from pytransit.orbits_f import orbits as of

from priors import UP,NP,JP,PriorSet

class RVLogPosteriorFunction(object):
    pass

class LCLogPosteriorFunction(object):
    def __init__(self, lcdata, tmodel=None, **kwargs):
        self.tmodel = tmodel or Gimenez()
        self.lcdata = lcdata

        self.group = None

        self.pid_ar = None
        self.pid_ld = None

        ## Precalculate the likelihood constants
        ## =====================================
        self.lln = -0.5*self.lcdata.npt * m.log(2*m.pi)

        ## Define priors
        ## =============
        pr = kwargs.get('priors', {})
        self.priors = []
        self.priors.extend([UP( 0.0005,    0.005, 'e', 'Average ptp scatter') for i in range(self.lcdata.npb)])
        self.priors.extend([UP(    0.0,      0.0, 'c', 'Contamination') for i in range(self.lcdata.npb)])
        self.ps = PriorSet(self.priors)

        self.np = len(self.priors)
        self.pv_start_idx = None


    def compute_continuum(self, pv):
        raise NotImplementedError

    def compute_transit(self, pv):
        z = of.z_eccentric(self.lcdata.time, *group._pv_o_physical, nthreads=0)
        f = self.tmodel(z, m.sqrt(pv[self.pid_ar[0]]), pv[self.pid_ld], 0.0)
        raise NotImplementedError

    def compute_lc_model(self, pv):
        raise NotImplementedError

    def log_posterior(self, pv):
        raise NotImplementedError
