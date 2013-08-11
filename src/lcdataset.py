import numpy as np
import matplotlib.pyplot as pl

class LCPassBand(object):
    def __init__(self, name, center, width, description='', kgroup=0):
        self.name = name
        self.center = center
        self.width = width
        self.description = description
        self.kgroup = kgroup

    def __str__(self):
        return "{:s} -- center: {:3.0f} nm  width: {:3.0f} nm".format(self.name, self.center, self.width)


class TSDataSet(object):
    """ Time series dataset"""

    def __init__(self, time):
        self.time = np.array(time)
        self.npt = self.time.size


class LCDataSet(TSDataSet):
    def __init__(self, time, lcdata, btime=0.3):
        """
        Parameters
          time   : Array of of exposure center times
          lcdata : Dictionary of flux and error arrays as {passband: (flux_pb, error_pb)}

        Optional parameters
          btime  : Minimum time gap between separate transits

        Notes
          The lcdata dictionary can either contain only the flux values, or both the fluxes
          and error estimates. If errors are not give, a constant average error of 1 ppt is 
          assumed.

          The btime parameter sets the time interval used to break the dataset into separate
          chunks.
        """

        super(LCDataSet, self).__init__(time)

        self.passbands = sorted(lcdata.keys(), key=lambda t: t.center)
        self.npb       = len(self.passbands)

        if len(lcdata[lcdata.keys()[0]]) == 2:
            self.fluxes = [np.asarray(lcdata[pb][0]) for pb in self.passbands]
            self.errors = [np.asarray(lcdata[pb][1]) for pb in self.passbands]
        else:
            self.fluxes = [np.asarray(lcdata[pb]) for pb in self.passbands]
            self.errors = [1e-3*np.ones_like(f) for f in self.fluxes]

        breaks = np.diff(self.time) > btime
        breaks = np.concatenate([[0], np.arange(self.npt)[breaks]+1, [self.npt]])
        self.transit_slices = [np.s_[breaks[i]:breaks[i+1]] for i in range(breaks.size-1)]


class RVDataSet(TSDataSet):
    def __init__(self, time, rv, err):
        super(RVDataSet, self).__init__(time)
        self.rv = rv
        self.err = err


if __name__ == '__main__':
    PB = LCPassBand

    times = np.concatenate([np.linspace(i,i+0.5,10) for i in range(4)])
    fluxes = {PB('z',500,100):np.concatenate([np.linspace(0,2,10) for i in range(4)]),
              PB('g',700,100):np.concatenate([np.linspace(0,2,10) for i in range(4)]),
              PB('b',400,100):np.concatenate([np.linspace(0,2,10) for i in range(4)])}

    ds = LCDataSet(times, fluxes)
