from scipy.ndimage import median_filter as mf
import numpy as np

from misc import fold

def find_jumps(date, flux, sigma=2.5, dx=4, minx=3, mfw=8, **kwargs):
    flux_mf  = mf(flux, mfw)
    dflux    = flux_mf[:-dx] - flux_mf[dx:]
    jump_indices = np.arange(date.size)[np.abs(dflux) > sigma*dflux.std()]

    if 'fj_rej_period' in kwargs.keys():
        jump_indices = [b for b,p in zip(jump_indices, fold(date[jump_indices], kwargs.get('fj_rej_period'), kwargs.get('fj_rej_center'), 0.5)) if abs(p-0.5)> kwargs.get('fj_rej_phase',0.03)] + [date.size-1]

    if len(jump_indices) > 0:
        ji_final = [0]
        i = 0
        finished = False

        while not finished:
            t = [jump_indices[i]]
            j = 1
            for j in range(1, len(jump_indices)-i):
                if jump_indices[i+j]-t[-1] < minx:
                    t.append(jump_indices[i+j])
                else:
                    break
            i += j
            ji_final.append(t[len(t)//2])
            if i >= len(jump_indices):
                finished = True
        jump_indices = ji_final + [flux.size-1]

    ## Plot the results if asked
    ## --------------------------
    if 'fj_axis' in kwargs.keys():
        ax = kwargs.get('fj_axis')
        ax.plot(date, flux, 'k.', alpha='0.25')
        for ji in jump_indices:
            ax.axvline(date[ji], c='k')
        ax.plot(date,flux_mf, 'w', lw=4)
        ax.plot(date,flux_mf, 'k', lw=2.5)
        
    return jump_indices


def label_areas(data, jumps, gap=4, **kwargs):
    labels = np.zeros(data.size, dtype=np.int32)
    lid = 1
    for i in range(len(jumps[:-1])):
        if(jumps[i+1]-jumps[i] > 2*gap):
            labels[jumps[i]+gap:jumps[i+1]-gap] = lid
            lid += 1
    return labels


def normalize_areas(flux, labels, gap=4, **kwargs):
    label_ids = np.unique(labels)
    f = flux.copy()
    for lid in label_ids[1:]:
        if lid > 1: 
            m_pre = m
        m = labels == lid
        if lid == 1:
            f[m] /= np.median(f[m])
        else:
            f[m] /= np.median(f[m][:20]) / np.median(f[m_pre][-20:])
    f /= np.median(f[labels!=0])
    f[labels==0] = mf(f, 10*gap)[labels==0]
    return f

def normalize(date, flux, jsigma=2.5, jdx=4, minx=3, mfw=8, gap=4, **kwargs):
    jumps = find_jumps(date,flux, jsigma, jdx, minx=minx, mfw=mfw, **kwargs)
    labels = label_areas(flux, jumps, gap)
    return normalize_areas(flux, labels, gap)
