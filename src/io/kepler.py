import numpy as np
import kplr

from ..keplerlc import KeplerLC

def load_kepler_lc(target, cadence='l', use_pdc=False, phase_offset=0, imin=0, imax=None, **kwargs):
    """
    Import a Kepler light curve.

    Parameters
    ----------

    koi        koi number
    planet     planet name (if koi not given)
    cadence    cadence to load
    use_pdc    load pdcsap flux instead of sap
    phase_offset
    imin
    imax

    bl_factor  baseline duration to include in transit durations
    tr_factor  in-transit duration to exclude for oot level estimation
    bl_min      minimum baseline duration
    """
    client = kplr.API()

    if isinstance(target, kplr.api.KOI):
        koi = target
    elif isinstance(target, str) and 'kepler-' in target.lower():
        koi = client.planet(planet).koi
    else:
        koi = client.koi(koi)

    bl_min    = kwargs.get('bl_min',    0.5)
    bl_factor = kwargs.get('bl_factor', 3.0)
    td_factor = kwargs.get('td_factor', 1.2)

    files = koi.get_light_curves(short_cadence=False if cadence == 'l' else True)
    cadence_string = 'llc' if cadence == 'l' else 'slc'

    t,f,q =[],[],[]
    for ff in files:
        if cadence_string in ff.filename:
            with ff.open() as hdul:
                fl = hdul[1]
                t.append(fl.data.field('time')) # + fl.header['bjdrefi'] + fl.header['bjdreff'])
                f.append(fl.data.field('pdcsap_flux'))
                q.append(hdul[0].header['quarter']*np.ones(t[-1].size, dtype=np.int))
    t,f,q = map(np.concatenate, (t,f,q))

    tcenter = koi.koi_time0bk
    period  = koi.koi_period
    trdur   = koi.koi_duration/24.

    return KeplerLC(t, f, tcenter+phase_offset*period, period, max(td_factor*trdur,0.18), max(bl_factor*trdur,0.5), q, **kwargs)
