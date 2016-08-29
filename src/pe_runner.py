from os.path import join, exists
from time import time
from pickle import dump, load

import numpy as np
import matplotlib.pyplot as pl
import emcee

from pyde.de import DiffEvol

class ProgressPlotter(object):
    def __init__(self, lpfun, name, pdir):
        self.f_hist  = pl.figure(figsize=[20,15], dpi=200)
        self.f_chain = pl.figure(figsize=[20,15], dpi=200)

        self.a_hist  = [self.f_hist.add_subplot(5, 5, i) for i in range(1, 23)]
        self.a_chain = [self.f_chain.add_subplot(5, 5, i) for i in range(1, 23)]

        self.pdir = pdir
        self.pids = range(lpfun.ps.ndim)
        self.name = name

    def update(self, chain, step):
        if step != 0:
            for aid, pid in enumerate(self.pids):
                self.a_hist[aid].cla()
                self.a_hist[aid].hist(chain[:,:step+1, pid].flat, range=np.percentile(chain[:,:step+1, pid].flat, [1,99]))
                self.a_chain[aid].cla()
                self.a_chain[aid].plot(chain[:,:step+1,pid].T, c='0', alpha=0.06, zorder=0)
                p = np.array(np.percentile(chain[:,:step+1,pid], [50,16,84], axis=0))
                self.a_chain[aid].plot(p.T, c='1', lw=3, zorder=100)
                self.a_chain[aid].plot(p.T, c='0', lw=2, zorder=101)

            self.f_hist.savefig(join(self.pdir,'progress_{}_hist.pdf'.format(self.name)), dpi=150)
            self.f_chain.savefig(join(self.pdir, 'progress_{}_chain.png'.format(self.name)), dpi=150)


def pe_runner(lpfun, basename, **kwargs):
    result_dir = kwargs.get('result_dir','.')
    plot_dir   = kwargs.get('plot_dir','.')
    runname    = kwargs.get('run_name','default')
    n_de_iters = kwargs.get('n_de_iterations',150)
    n_mc_iters = kwargs.get('n_mc_iterations',150)
    pop_size   = kwargs.get('pop_size',100)
    thinning   = kwargs.get('thinning',1)
    mc_continue= kwargs.get('mc_continues',True)
    update_interval = kwargs.get('update_interval',60)

    do_de = kwargs.get('do_de', False)
    do_mc = kwargs.get('do_mc', False)

    pplot   = ProgressPlotter(lpfun, '{:s}_{:s}'.format(basename,runname), plot_dir)

    de_res_fname = join(result_dir,'{:s}_{:s}_de.pkl'.format(basename,runname))
    mc_res_fname = join(result_dir,'{:s}_{:s}_mc.pkl'.format(basename,runname))
    mc_bck_fname = join(result_dir,'{:s}_{:s}_mc_back.pkl'.format(basename,runname))

    de_res_exists = exists(de_res_fname)
    mc_res_exists = exists(mc_res_fname)

    ## ================================================================================
    ## DE - Run Differential Evolution
    ## ================================================================================
    if do_de or not de_res_exists:
        t_start = time()
        de = DiffEvol(lambda pv: -lpfun(pv), lpfun.ps.bounds, pop_size)
        best_ll = 1e80
        for i,res in enumerate(de(n_de_iters)):
            if res[1] < best_ll:
                print (i, res[1])
                best_ll = res[1]

        with open(de_res_fname, 'w') as fout:
            dump({'population':de.population, 'best':de.minimum_location}, fout)
        de_population, de_best_fit = de.population, de.minimum_location
        print ("time taken: {:4.2f}".format(time()-t_start))
    else:
        with open(de_res_fname, 'r') as fin:
            de_res = load(fin)
        de_population, de_best_fit = de_res['population'], de_res['best']

    if not do_mc and not mc_res_exists:
        return de_population, de_best_fit

    ## ================================================================================
    ## MCMC - Run emcee
    ## ================================================================================
    if do_mc or not mc_res_exists:
        t_iteration_start = time()
        t_last_update = time()
        t_start = time()

        if mc_res_exists and mc_continue:
            print ('Continuing from the previous MCMC run.')
            with open(mc_res_fname,'r') as fin:
                pv0 = load(fin)['chain'][:,-1,:]
        else:
            pv0 = de_population

        if pv0.shape[0] > pop_size:
            pv0 = pv0[:pop_size,:].copy()
        elif pv0.shape[0] < pop_size:
            pv0 = np.tile(pv0, [np.ceil(pop_size/pv0.shape[0]), 1])[:pop_size, :].copy()

        #pv0[:,14] = np.random.uniform(0.01, 0.98, size=300)
        #m = pv0[:,2] > 0.03
        #pv0[m,:] = pv0[~m,:][:m.sum(),:]
        #pv0[:,2] = np.random.uniform(0.005, 0.1, size=300)
        #print pv0.shape;exit()

        sampler = emcee.EnsembleSampler(pop_size, lpfun.ps.ndim, lpfun)
        for i, e in enumerate(sampler.sample(pv0, iterations=n_mc_iters, thin=thinning)):
            t_cur = time()
            print ("{:4d}/{:4d}  Secs/iteration {:6.2f}  Last update {:6.2f} s ago  Total time {:6.2f} s   Acceptance {:6.3f}".format(i+1, n_mc_iters, t_cur-t_iteration_start, t_cur-t_last_update, t_cur-t_start, sampler.acceptance_fraction.mean()))
            if i != 0 and (t_cur - t_last_update > update_interval):
                pplot.update(sampler.chain, i//thinning)
                t_last_update = t_cur
                with open(mc_bck_fname,'w') as fout:
                    dump({'chain':sampler.chain, 'logl':sampler.lnprobability}, fout)
            t_iteration_start = t_cur

        with  open(mc_res_fname, 'w') as fout:
            dump({'chain':sampler.chain, 'logl':sampler.lnprobability}, fout)
        chain = sampler.chain

    else:
        with open(mc_res_fname, 'r') as fin:
            chain = load(fin)['chain']

    return chain
