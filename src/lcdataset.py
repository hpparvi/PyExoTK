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
    def __init__(self, time, fluxes, btime=0.3):
        """
        Parameters
          times  : Array of of exposure center times
          fluxes : Dictionary of fluxe arrays as {(pb_name,pb_center,pb_width): flux_pb}

        Optional parameters
          btime  : Minimum time gap between separate transits
        """

        super(LCDataSet, self).__init__(time)

        self.passbands = sorted(fluxes.keys(), key=lambda t: t.center)
        self.fluxes    = [np.asarray(fluxes[pb]) for pb in self.passbands]
        self.npb       = len(self.passbands)


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
