import math  as m
import numpy as np
from time import time

from numpy import square

from transitlightcurve.transitmodel import gimenez_f
from transitlightcurve.utilities_f import utilities as uf
from transitlightcurve.orbits import orbits
from transitlightcurve.fitting.mcmcprior import UP, GP, JP
from transitlightcurve.fitting.priorset import PriorSet

from lpfunction import LogPosteriorFunction

from scipy import constants as cn

PL1,PL2 = 2*cn.h*cn.c**2, cn.h*cn.c

class BlendLogPosteriorFunction(LogPosteriorFunction):
    def __init__(self, lc_data, priors, filter_names, filter_centers, nwalkers=150, npol=100, nthreads=2):
        super(BlendLogPosteriorFunction, self).__init__(lc_data, nwalkers, npol, nthreads)

        self.n_colors = len(lc_data)
        self.filter_names = filter_names
        self.filter_centers = filter_centers

        pr = priors
        self.priors = [pr.get('tc'),                                                                       ##  0
                       pr.get( 'p'),                                                                       ##  1
                       pr.get('k2', UP(0.07**2, 0.16**2, 'k2', 'Squared radius ratio', 'sqrt(1/Rs)')),     ##  2
                       pr.get('it', UP(     10,      50, 'it', '2 / transit duration', '1/d')),            ##  3
                       pr.get( 'b', UP(      0,    0.99,  'b', 'impact parameter')),                       ##  4
                       pr.get( 'c', UP(      0,   0.001,  'c', 'First channel contamination')),            ##  5
                       pr.get('T3', UP(   2000,    6000, 'T3', 'Third source temperature', 'K')),          ##  6
                       pr.get( 'e', UP(      0,   0.001,  'e', 'Eccentricity')),                           ##  7
                       pr.get( 'w', UP(      0,   0.001,  'w', 'Argument of periastron'))]                 ##  8

        [self.priors.extend([UP(       0,          1.3,  'u', 'Linear limb darkening coefficient'),        ##  9 + 2*i_c
                             UP(    -0.3,          0.7,  'v', 'Quadratic limb darkening coefficient')]     ## 10 + 2*i_c
                            ) for i in range(self.n_colors)]

        self.priors.extend([UP(   0.1*e,    10*e, 'e', 'Mean error') for e in self.flux_e])                ##  9 + 2*n_c + i_c
        self.priors.extend([UP(  1-1e-3,  1+1e-3, 'zp', 'Zeropoint') for i in range(self.n_colors)])       ##  9 + 3*n_c + i_c

        self.priors

        self.ps = PriorSet(self.priors)

        ## Add extra priors
        ## ================
        self.prior_t14  = priors.get( 'T14', None)
        self.prior_rho  = priors.get( 'rho', None)
        self.prior_logg = priors.get('logg', None)

        self.upd = [True]+(self.n_colors-1)*[False]
        self.cid = range(self.n_colors)

        self.ld_start = 9
        self.ep_start = self.ld_start + 2*self.n_colors
        self.zp_start = self.ep_start +   self.n_colors

        self.ld_sl = [np.s_[self.ld_start+2*ic:self.ld_start+2*(ic+1)] for ic in self.cid]
        self.er_sl = np.s_[self.ep_start:self.ep_start+self.n_colors]


    def Planck(self, T, wl):
        return PL1/wl**5 / (m.exp(PL2/(wl*cn.k*T))-1)

    def Planck_a(self, T, wl):
        return PL1/wl**5 / (np.exp(PL2/(wl*cn.k*T))-1)

    def contamination_from_ratios(self, c0, l0, l1, T):
        """Returns the contamination on wavelength l1 given the source temperature T and contamination c0 on wavelength l0"""
        B0   = self.Planck_a(T, l0)
        B1   = self.Planck_a(T, l1)
        return  np.concatenate([[c0],c0*(B1/B0)])


    def get_contamination(self, pv):
        return self.contamination_from_ratios(pv[5], self.filter_centers[0], self.filter_centers[1:], pv[6])


    def get_ew(self, e, w):
        return e, w


    def compute_continuum(self, pv):
        return [pv[self.zp_start + ic] for ic in self.cid]


    def compute_transit(self, pv, _i=None, _a=None, _k=None):
        _i = _i or uf.i_from_bitpew(pv[4], pv[3], pv[1], pv[7], pv[8])
        _a = _a or uf.a_from_bitpew(pv[4], pv[3], pv[1], pv[7], pv[8])
        _k = _k or m.sqrt(pv[2])

        cns =  self.contamination_from_ratios(pv[5], self.filter_centers[0], self.filter_centers[1:], pv[6])

        zs = [orbits.z_eccentric(time, pv[0], pv[1], _a, _i, pv[7], pv[8], nthreads=self.nthreads) for time in self.time]
        f  = [self.tmodel(z, _k, pv[self.ld_sl[ic]], cns[ic]) for z,ic in zip(zs, self.cid)]

        return f


    def compute_lc_model(self, pv, _i=None, _a=None, _k=None):
        return [c*t for c,t in zip(self.compute_continuum(pv), self.compute_transit(pv, _i, _a, _k))]


    def __call__(self, pv):
        ## Do a boundary test to check if we need to calculate anything further
        ## ====================================================================
        ld_sums = np.array([pv[self.ld_sl[ic]].sum() for ic in self.cid])
        if np.any(pv < self.ps.pmins) or np.any(pv>self.ps.pmaxs) or np.any(ld_sums < 0) or np.any(ld_sums > 1):
            return -1e18

        ## Calculate basic parameter priors
        ## ================================
        log_prior  = self.ps.c_log_prior(pv)
        
        ## Precalculate the orbit parameters in order to reduce the unnceressary computations
        ## ===================================================================================
        _i = uf.i_from_bitpew(pv[4], pv[3], pv[1], pv[7], pv[8])
        _a = uf.a_from_bitpew(pv[4], pv[3], pv[1], pv[7], pv[8])
        _k = m.sqrt(pv[2])

        ## Calculate derived parameter priors
        ## ==================================
        log_dpriors = 0.

        if self.prior_t14:
            t14 = orbits.duration_eccentric_w(pv[1], _k, _a, _i, pv[7], pv[8], 1)
            log_dpriors += self.prior_t14.log(t14)

        ## Calculate chi squared values
        ## ============================
        flux_m     = self.compute_lc_model(pv, _i, _a, _k)
        chisqr_lc  = [uf.chi_sqr_se(fo, fm, err) for fo, fm, err in zip(self.flux_o, flux_m, pv[self.er_sl])]

        ## Return the log posterior
        ## ========================
        return log_prior + log_dpriors + self.lln + sum([- npt*m.log(e) - chisqr for npt,e,chisqr in zip(self.npt, pv[self.er_sl], chisqr_lc)])


    def create_distributions(self, fc):
        from transitlightcurve.utilities import stellar_density, BIC

        kd = np.sqrt(fc[:,2])
        ed = fc[:,7]
        wd = fc[:,8]
        ad = np.array([uf.a_from_bitpew(_b, _it, _p, _e, _w) for _b, _it, _p, _e, _w in zip(fc[:,4], fc[:,3], fc[:,1], ed, wd)])
        Id = np.array([uf.i_from_bitpew(_b, _it, _p, _e, _w) for _b, _it, _p, _e, _w in zip(fc[:,4], fc[:,3], fc[:,1], ed, wd)]) * 180/np.pi
        dd = np.array([orbits.duration_eccentric_w(_p,_k,_a,_i,_e,_w,1) for _p,_k,_a,_i,_e,_w in zip(fc[:,1], kd, ad, Id*np.pi/180., ed, wd)])
        rd = np.array([stellar_density(_d,_p,_k,_b,_e,_w, False) for _d,_p,_k,_b,_e,_w in zip(dd*24*60*60, fc[:,1]*24*60*60, kd, fc[:,4], ed, wd)])

        p_labels = ['Zero epoch', 'Period', 'T14', 'T14', 'Radius ratio', 'Scaled semi-major axis', 'Inclination', 'Impact parameter',
                    'Eccentricity', 'Argument of periastron', 'Stellar density', 'Third source temperature', ]
        p_units = ['HJD', 'd', 'd', 'h', 'Rs', 'Rs', 'deg', '-', '-', 'rad', 'g/cm^3', 'K']
        p_dists  = [fc[:,0], fc[:,1], dd, 24*dd, kd, ad, Id, fc[:,4], ed, wd, rd, fc[:,6]]

        [p_labels.extend(['Linear ldc {}'.format(cid), 'Quadratic ldc {}'.format(cid)]) for cid in self.cid]
        [p_units.extend(['-', '-']) for cid in self.cid]
        [p_dists.extend([fc[:,self.ld_start+2*cid], fc[:,self.ld_start+2*cid+1]]) for cid in self.cid]

        p_labels.extend(['{} contamination'.format(f) for f in self.filter_names])
        p_units.extend(['-' for i in self.cid])
        p_dists.extend(np.hsplit(np.array([self.get_contamination(pv) for pv in fc]), self.n_colors))

        p_labels.extend(['Error {}'.format(cid) for cid in self.cid])
        p_units.extend(['mmag' for cid in self.cid])
        p_dists.extend([1e3*fc[:,self.ep_start+cid] for cid in self.cid])

        p_labels.extend(['Zeropoint {}'.format(cid) for cid in self.cid])
        p_units.extend(['' for cid in self.cid])
        p_dists.extend([fc[:,self.zp_start+cid] for cid in self.cid])

        p_ids    = range(len(p_dists))

        return p_labels, p_units, p_dists, p_ids
