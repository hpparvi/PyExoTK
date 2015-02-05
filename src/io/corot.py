"""Contains CoRoT specific constants."""
from __future__ import division

import sys
import os.path
import numpy as np
import scipy.constants as c
import pyfits as pf
from itertools import product
from numpy import array

## --- CoRoT specific constants ---
##

orbit_period = 0.071458  # The period around the Earth
time_origin  = 2451545.  # The epoch from where the CoRoT time starts

channel_center = {'r':0.800e-6, 'g':0.650e-6, 'b':0.550e-6, 'w':0.700e-6}
channel_fits_names = {'r':'red', 'g':'green', 'b':'blue', 'w':'white'}

try:
    corot_dir = os.environ['COROT_DATA_PATH']
except:
    corot_dir = None
    logging.warning("COROT_DATA_PATH is not set")

class CoRoT_target:
    def __init__(self, fname, name, published_parameters=None, stellar_parameters=None, contamination=0.0, **kwargs):
        self.file = fname
        self.name = name
        self.basename = name.replace('-','_')
        self.published_parameters = published_parameters
        self.stellar_parameters = stellar_parameters
        self.contamination = contamination
        self.colors = '_CHR_' in fname or '_IMAG' in fname

        pb = self.published_parameters

        if pb is not None:
            pb['transit duration'] = np.array(pb['transit duration [h]']) / 24
            pb['T14']              = pb.get('transit duration')

            self.planet_period     = pb.get('period')[0]
            self.transit_center    = pb.get('transit center')[0]
            self.transit_duration  = pb.get('transit duration')[0]
            self.transit_width     = self.transit_duration
            self.radius_ratio      = pb.get('radius ratio', [None])[0]
            self.semimajor_axis    = pb.get('semi-major axis')

            self.p  = self.planet_period
            self.tc = self.transit_center
            self.k  = self.radius_ratio
            self.sp = self.stellar_parameters
            self.a  = self.semimajor_axis

    def get_uniform_limits(self, name, s=1):
        v = self.published_parameters.get(name)

        defaults = {'omega':[0.,360.], 'inclination':[88, 90]}

        if v is None:
            l = array(defaults[name])
        elif len(v) == 1:
            l = array([0., v[0]])
        elif len(v) == 2:
            l = array([v[0]-s*v[1], v[0]+s*v[1]])
        else:
            l = array([v[0]-s*v[1], v[0]+s*v[2]])

        if name == 'omega':
            l = np.clip(l, 0, 360)
        elif name in ['eccentricity', 'radius ratio', 'impact parameter']:
            l = np.clip(l, 0, 1)
        elif name in ['inclination']:
            l = np.clip(l, 0, 90)

        return l


def import_as_MTLC(ctarget, duration=0.2, tduration=None, maxpts=None, **kwargs):
    """
    Import a CoRoT light curve as list of MultiTransiLightCurves.

    Parameters
      ctarget            CoRoT_target   The CoRoT planet to import
      width              [days]   Included duration of the transit centered light curve
      twidth             [days]   Transit duration
      maxpts             int      Maximum number of points to import

    Keyword arguments
      combine_channels   Boolean  Should the RGB channels be merged
      phase_offset       [0..1]   In the case you might want to import something
                                  else than the transit (mainly for secondaries)
      channels           [r,g,b]  Channels to import
      cadences           [s,l]    Time cadences to import


    """
    hdu_d = pf.open(ctarget.file)
    dat_d = hdu_d[1].data

    twidth = tdur or ctarget.transit_duration
    maxpts = maxpts or -1 

    combine_channels = kwargs.get('combine_channels', False)
    phase_offset     = kwargs.get('phase_offset',       0.0)
    channels         = kwargs.get('channels', ['r','g','b'])
    cadences         = kwargs.get('cadences', ['l','s'])
    transit_center   = kwargs.get('transit_center', ctarget.transit_center)

    ch_ids, cd_ids   = {'r':0,'g':2,'b':4}, {'l':0,'s':1}

    stat = dat_d.field('STATUS')
    maskOutOfRange = np.bitwise_and(stat, 1) == 0 if kwargs.get('mask_oor', True) else np.ones(stat.size, np.bool)
    maskOverSAA    = np.bitwise_and(stat, 4) == 0 if kwargs.get('mask_saa', True) else np.ones(stat.size, np.bool)
    mask           = (np.logical_and(maskOverSAA, maskOutOfRange))

    date = dat_d.field('DATEJD')[mask].copy().astype(np.float64)

    is_imag = '_imag' in ctarget.file.lower()

    flux = []; fdev = []
    if not ctarget.colors:
        channels = ['w']
        flux.append(dat_d.field('whiteflux')[mask][:maxpts].copy().astype(np.float64))
        fdev.append(dat_d.field('whitefluxdev')[mask][:maxpts].copy().astype(np.float64))
    else:
        if not combine_channels:
            for ch in channels:
                if is_imag:
                    flux.append(dat_d.field( channel_fits_names[ch]+'flux_imag')[mask][:maxpts].copy().astype(np.float64))
                    fdev.append(np.zeros(flux[-1].size))
                else:
                    flux.append(dat_d.field( channel_fits_names[ch]+'flux')[mask][:maxpts].copy().astype(np.float64))
                    fdev.append(dat_d.field( channel_fits_names[ch]+'fluxdev')[mask][:maxpts].copy().astype(np.float64))
        else:
            channels = ['w']
            flux.append(dat_d.field('redflux')[mask][:maxpts].copy().astype(np.float64)+
                        dat_d.field('greenflux')[mask][:maxpts].copy().astype(np.float64)+
                        dat_d.field('blueflux')[mask][:maxpts].copy().astype(np.float64))

            fdev.append(np.sqrt(dat_d.field('redfluxdev')[mask][:maxpts].copy().astype(np.float64)**2+
                        dat_d.field('greenfluxdev')[mask][:maxpts].copy().astype(np.float64)**2+
                        dat_d.field('bluefluxdev')[mask][:maxpts].copy().astype(np.float64)**2))
                
    expmask         = fdev[-1] < 1e-7
    date[expmask]  -= 16./(60.*60.*24.)
    date[~expmask] += 234./(60.*60.*24.)
    date           += time_origin

    cadence_exists = {'l':np.any(~expmask), 's':np.any(expmask)}
    cadence_mask   = {'l':~expmask, 's':expmask}

    ft, et, dt = {}, {}, {}
    for c,ce in cadence_exists.items():
        if ce:
            for fl,fd,ch in zip(flux,fdev,channels):
                ft[c,ch] = fl[cadence_mask[c]]
                et[c,ch] = fd[cadence_mask[c]]
            dt[c] = date[cadence_mask[c]]
    date, flux, fdev = dt, ft, et

    return date, flux, fdef
