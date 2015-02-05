import math as m
import numpy as np

from numpy import any, asarray, zeros, zeros_like, s_, pi

from pytransit import Gimenez, MandelAgol
from pytransit.orbits_f import orbits as of
from exotk.priors import UP,NP,PriorSet
from exotk.utils.orbits import as_from_rhop, a_from_rhoprs, i_from_baew

class SingleTransitMultiColorQuadraticLPF(object):
    def __init__(self, time, flux, airmass, tcenter, tduration, priors, nthr=4, tmodel=None):
        self.tmodel = tmodel or MandelAgol(nthr=nthr, lerp=True)
        self.time = asarray(time)
        self.flux = asarray(flux)
        self.airmass = asarray(airmass)
        self.npb = self.flux.shape[1]
        self.mask_oot = (self.time < tcenter-0.55*tduration) | (self.time > tcenter+0.55*tduration)
        self.ld = zeros(2*self.npb)
        
        self.baseline = zeros_like(self.flux)
        self.tmparr = zeros_like(self.flux)
        
        elc = [f[self.mask_oot].std() for f in self.flux.T]
        self.priors = [priors['transit_center'],                                ##  0 transit center, required
                       priors['period'],                                        ##  1 period, required
                       priors['stellar_density'],                               ##  2 stellar density, required
                       priors.get('impact_parameter', UP(0.0, 0.99, 'b'))]      ##  3 impact parameter
        
        kmin, kmax = priors['radius_ratio'].limits()
        for i in range(self.npb):
            self.priors.append(UP(kmin**2, kmax**2, 'k2_{}'.format(i)))         ##  as + i       squared radius ratios
                
        for i in range(self.npb):
            self.priors.append(UP(0.15*elc[i], 2*elc[i],  'el_{}'.format(i)))   ##  es + i       point-to-point scatter

        for i in range(self.npb):
            self.priors.extend([UP(     0.0001,        1,   'u_{}'.format(i)),  ##  ls + 0 + 2*i linear ldc
                                UP(     0.0001,        1,   'v_{}'.format(i))]) ##  ls + 1 + 2*i quadratic ldc

        blmin, blmax = priors.get('baseline', UP(0.99,1.01)).limits()
        for i in range(self.npb):
            self.priors.extend([UP(      blmin,    blmax,  'bl_{}'.format(i)),  ##  bs + 0 + 2*i baseline level
                                UP(      0.000,    0.020,  'am_{}'.format(i))]) ##  bs + 1 + 2*i airmass correction

        self.ps = PriorSet(self.priors)

        ks = self.k2_start = 4
        es = self.err_start = ks+self.npb 
        ls = self.ldc_start = es+self.npb
        bs = self.baseline_start = ls + 2*self.npb
        
        self.k2_slice   = s_[ks:es]
        self.err_slice  = s_[es:ls]
        self.ldc_slice  = s_[ls:bs]
        self.bsl_slices = [s_[bs+2*i:bs+2*(i+1)] for i in range(self.npb)]
        
        
    def compute_lc_model(self, pv):
        _tc, _p, _b = pv[0], pv[1], pv[3]
        _ld = pv[self.ldc_slice]
        _a = as_from_rhop(pv[2], pv[1])
        _i = m.acos(_b/_a)

        self.ld[0::2] = 2.*np.sqrt(_ld[0::2])*_ld[1::2]
        self.ld[1::2] = np.sqrt(_ld[0::2])*(1.-2.*_ld[1::2])
        
        k = np.sqrt(pv[self.k2_slice])
        _k = k.mean()
        kf = (k/_k)**2
        
        z = of.z_circular(self.time, _tc, _p, _a, _i, 4)
        flux = self.tmodel(z, _k, self.ld)
        
        return kf*(flux-1.)+1.

    def compute_baseline(self, pv):
        for i in range(self.npb):
            bpv = pv[self.bsl_slices[i]]
            self.baseline[:,i] = bpv[0]/np.exp(bpv[1]*self.airmass)
        return self.baseline

    def normalize_flux(self, pv):
        return self.flux/self.compute_baseline(pv)
    
    def log_posterior(self, pv):
        if any(pv < self.ps.pmins) or any(pv>self.ps.pmaxs): return -1e18
        o = self.normalize_flux(pv)
        m = self.compute_lc_model(pv)
        self.tmparr[:] = ((o-m)/pv[self.err_slice])**2
        chi_sqr = self.tmparr.sum()
        log_p   = self.ps.c_log_prior(pv)
        log_l   = - 0.5*self.time.size*np.log(2*pi) - (self.time.shape[0]*np.log(pv[self.err_slice])).sum() - 0.5*chi_sqr
        return log_p + log_l
