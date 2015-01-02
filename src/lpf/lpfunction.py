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
 

        self.pid_private = None
        ## Precalculate the likelihood constants
        ## =====================================
        self.lln = -0.5*self.lcdata.npt * m.log(2*m.pi)

        ## Define priors
        ## =============
        pr = kwargs.get('priors', {})
        self.priors = []
        self.priors.extend([UP(0.2, 5.0, 'e', 'Error multiplier') for i in range(self.lcdata.npb)])
        self.priors.extend([UP(0.0, 0.01, 'c', 'Contamination') for i in range(self.lcdata.npb)])
        self.ps = PriorSet(self.priors)

        self.np = len(self.priors)
        self.pv_start_idx = None


    def compute_continuum(self, pv):
        raise NotImplementedError


    def compute_transit(self, pv):
        print self.group._pv_o_physical
        z = of.z_eccentric(self.lcdata.time, *self.group._pv_o_physical, nthreads=0)
        f = self.tmodel(z, m.sqrt(pv[self.pid_ar[0]]), pv[self.pid_ld], 0.0)
        return f


    def compute_lc_model(self, pv):
        return self.compute_transit(pv)


    def log_posterior(self, pv):
        log_prior  = self.ps.c_log_prior(pv[self.pid_private])

        flux_m  = self.compute_lc_model(pv)
        err     = self.lcdata.errors*pv[self.pid_err]
        chisqr  = (((self.lcdata.fluxes - flux_m)/err)**2).sum()

        log_likelihood = self.lln - 0.5*(np.log(err)).sum() -0.5*chisqr

        return log_prior + log_likelihood

