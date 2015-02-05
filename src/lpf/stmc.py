import math as m
import numpy as np

from numpy import any, asarray, atleast_2d, zeros, zeros_like, s_, pi
from numpy.random import uniform

from pytransit import Gimenez, MandelAgol
from pytransit.orbits_f import orbits as of
from exotk.priors import UP,NP,PriorSet
from exotk.utils.orbits import as_from_rhop, a_from_rhoprs, i_from_baew

class SingleTransitMultiColorLPF(object):
    def __init__(self, time, flux, airmass, tcenter, tduration, priors, nthr=4, tmodel=None, verbosity=0):
        self.tmodel = tmodel or MandelAgol(nthr=nthr, lerp=False)
        self.nldc = tmodel.nldc
        self.time = asarray(time)
        self.flux = atleast_2d(flux)
        self.airmass = asarray(airmass)
        self.npt = self.flux.shape[0]
        self.npb = self.flux.shape[1]
        self.mask_oot = (self.time < tcenter-0.55*tduration) | (self.time > tcenter+0.55*tduration)
        self.ld = zeros(2*self.npb)
        self.informative_limb_darkening = False
        self.verbosity = verbosity

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
            for j in range(self.nldc):
                ldc_label =  'ldc_{:d}_{:d}'.format(i,j)
                if ldc_label in priors.keys():
                    if self.verbosity > 0:
                        print 'Using an informative prior on limb darkening: ' + ldc_label
                    self.informative_limb_darkening = True

                self.priors.append(priors.get(ldc_label, UP(-0.5, 1, ldc_label)))  ##  ls + 0 + nldc*i ldc coefficients

        blmin, blmax = priors.get('baseline', UP(0.99,1.01)).limits()
        for i in range(self.npb):
            self.priors.extend([UP(      blmin,    blmax,  'bl_{}'.format(i)),  ##  bs + 0 + 2*i baseline level
                                UP(     -0.020,    0.020,  'am_{}'.format(i))]) ##  bs + 1 + 2*i airmass correction

        self.ps = PriorSet(self.priors)

        ks = self.k2_start = 4
        es = self.err_start = ks+self.npb 
        ls = self.ldc_start = es+self.npb
        bs = self.baseline_start = ls + self.nldc*self.npb
        
        self.k2_slice   = s_[ks:es]
        self.err_slice  = s_[es:ls]
        self.ldc_slice  = s_[ls:bs]
        self.bsl_slices = [s_[bs+2*i:bs+2*(i+1)] for i in range(self.npb)]
        
        
    def create_initial_population(self, npop):
        pv0 = self.ps.generate_pv_population(npop)
        if not self.informative_limb_darkening:
            pv0[:, self.ldc_slice] = uniform(0.0, 1./self.nldc, size=(npop, self.nldc*self.npb))
        return pv0

    def compute_lc_model(self, pv):
        _tc, _p, _b = pv[0], pv[1], pv[3]
        _ld = pv[self.ldc_slice]
        _a = as_from_rhop(pv[2], pv[1])
        _i = m.acos(_b/_a)
        
        k = np.sqrt(pv[self.k2_slice])
        _k = k.mean()
        kf = (k/_k)**2
        
        z = of.z_circular(self.time, _tc, _p, _a, _i, 4)
        flux = self.tmodel(z, _k, pv[self.ldc_slice])
        
        return (kf*(flux-1.)+1.).reshape((self.npt,self.npb))


    def compute_baseline(self, pv):
        for i in range(self.npb):
            bpv = pv[self.bsl_slices[i]]
            self.baseline[:,i] = bpv[0]/np.exp(bpv[1]*self.airmass)
        return self.baseline


    def normalize_flux(self, pv):
        return self.flux/self.compute_baseline(pv)

    
    def log_posterior(self, pv):
        if any(pv < self.ps.pmins) or any(pv>self.ps.pmaxs): return -1e18
        #if not (0. < pv[self.ldc_slice].sum() < 1.): return -1e18

        o = self.normalize_flux(pv)
        m = self.compute_lc_model(pv)
        self.tmparr[:] = ((o-m)/pv[self.err_slice])**2
        chi_sqr = self.tmparr.sum()
        log_p   = self.ps.c_log_prior(pv)
        log_l   = - 0.5*self.time.size*np.log(2*pi) - (self.time.shape[0]*np.log(pv[self.err_slice])).sum() - 0.5*chi_sqr
        return log_p + log_l
