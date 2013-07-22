from __future__ import division

import math
import numpy as np
from scipy.constants import G

from priors import UP,NP,JP,PriorSet
from lpf.lpfunction import LCLogPosteriorFunction, RVLogPosteriorFunction

AC   = (G/(3*np.pi))**(1/3)
DTOS = 86400

class LPFGroup(object):

    def __init__(self, orbit_priors={}):
        self.lc_lpfunctions = []
        self.rv_lpfunctions = []

        self.lclp = self.lc_lpfunctions
        self.rvlp = self.rv_lpfunctions

        pr = orbit_priors
        self.orbit_priors = [pr.get( 'tc'),                                                            ##  0  - Transit center
                             pr.get(  'p'),                                                            ##  1  - Period
                             pr.get('rho', UP(    0.5,     2.0, 'rho', 'stellar density', 'g/cm^3')),  ##  2  - Stellar density
                             pr.get(  'b', UP(      0,    0.99,   'b', 'impact parameter')),           ##  3  - Impact parameter
                             pr.get(  'e', UP(      0,   0.001,   'e', 'Eccentricity')),               ##  4  - Eccentricity
                             pr.get(  'w', UP(      0,   0.001,   'w', 'Argument of periastron'))]     ##  5  - Argument of periastron
        self.orbit_ps = PriorSet(self.orbit_priors)
        self.lc_passbands = []

        self.nor = 6
        self.nar = None
        self.nld = None
        self.nlc = None
        self.nrv = None

        self.ar_start_idx = self.nor
        self.ld_start_idx = None
        self.lc_start_idx = None
        self.rv_start_idx = None


    def add_lpfunction(self, lpfunction):
        lpfunction.group = self

        if isinstance(lpfunction, LCLogPosteriorFunction):
            self.lc_lpfunctions.append(lpfunction)
            self.lc_passbands.extend(lpfunction.lcdata.passbands)
        elif isinstance(lpfunction, RVLogPosteriorFunction):
            self.rv_lpfunctions.append(lpfunction)
        else:
            raise NotImplementedError


    def finalize(self):
        ## Optimise the area ratio access
        ## ==============================
        ar_groups =  list(np.unique(np.concatenate([[pb.kgroup for pb in lpf.lcdata.passbands] for lpf in self.lclp])))
        for lpf in self.lclp:
            lpf.pid_ar = [self.ar_start_idx + ar_groups.index(pb.kgroup) for pb in  lpf.lcdata.passbands]
        self.nar = len(ar_groups)
        self.ld_start_idx = self.ar_start_idx + self.nar

        ## Optimise the limb darkening coefficient access
        ## ==============================================
        lds = self.ld_start_idx
        self.lclp[0].pid_ld = np.s_[lds:lds+2*self.lclp[0].lcdata.npb]
        final_pb_set = self.lc_passbands[:self.lclp[0].lcdata.npb]
        for i,lpf in enumerate(self.lc_lpfunctions[1:]):
            lp_indices = []
            for j,pb in enumerate(lpf.lcdata.passbands):
                try:
                    lp_indices.append(final_pb_set.index(pb))
                except ValueError:
                    final_pb_set.append(pb)
                    lp_indices.append(len(final_pb_set)-1)

            if np.all(np.diff(lp_indices) == 1):
                lpf.pid_lc = np.s_[lds+lp_indices[0] : lds+2*(lp_indices[-1]+1)]
            else:
                lpf.pid_lc = np.concatenate([(lds+2*li, lds+2*li+1) for li in lp_indices])

        self.lc_passbands = final_pb_set
        self.nld = 2*len(self.lc_passbands)

        ## Setup the private space for logposterior functions
        ## ==================================================
        self.lc_start_idx = self.ld_start_idx+self.nld+np.cumsum([0]+[lpf.np for lpf in self.lclp[:-1]])        
        for i,lpf in enumerate(self.lclp):
            lpf.pv_start_idx = self.lc_start_idx[i]
        self.nlc = sum([lpf.np for lpf in self.lclp])


    def map_orbit(self, pv):
        a = AC*(pv[1]**2*DTOS * 1e3*pv[2])**(1/3)
        i = math.acos(pv[4]/a)
        self._pv_o_physical = pv[:6].copy()
        self._pv_o_physical[2] = a
        self._pv_o_physical[3] = i
        return self._pv_o_physical


    def __call__(self, pv):
        self.map_orbit(pv)
        return self.orbit_ps.c_log_prior(pv[0:7])
    

if __name__ == '__main__':
    from lcdataset import LCPassBand, LCDataSet

    PB = LCPassBand

    pbs = [PB('z',500,100), PB('g',700,100, kgroup=1), PB('b',400,100), PB('h', 1200, 100)]

    times1 = np.concatenate([np.linspace(i,i+0.5,10) for i in range(4)])
    fluxes1 = {pbs[0]:np.concatenate([np.linspace(0,2,10) for i in range(4)]),
               pbs[1]:np.concatenate([np.linspace(0,2,10) for i in range(4)]),
               pbs[2]:np.concatenate([np.linspace(0,2,10) for i in range(4)])}

    times2 = np.concatenate([np.linspace(i+8,i+8.5,10) for i in range(4)])
    fluxes2 = {pbs[0]:np.concatenate([np.linspace(0,2,10) for i in range(4)]),
               pbs[3]:np.concatenate([np.linspace(0,2,10) for i in range(4)]),
               pbs[2]:np.concatenate([np.linspace(0,2,10) for i in range(4)])}

    ds1 = LCDataSet(times1, fluxes1)
    ds2 = LCDataSet(times2, fluxes2)

    lpf1 = LCLogPosteriorFunction(ds1)
    lpf2 = LCLogPosteriorFunction(ds2)

    priors = {'tc': NP(0,0.1, 'tc'),
              'p':  NP(0,1.0, 'p')}

    g = LPFGroup(priors)
    g.add_lpfunction(lpf1)
    g.add_lpfunction(lpf2)
    g.finalize()
