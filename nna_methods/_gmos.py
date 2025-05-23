__all__ = ['GMOS']
from ._nna import NNA
import numpy as np


class GMOS(NNA):
    def __init__(self, rs=None, to_meter=1):
        """
        Gridded Model Output Statistics (GMOS) is a relatively simple objective
        analysis method described by Glahn (2009 and 2012). The basic method
        has been implemented, but many advanced options have not.

        Arguments
        ---------
        rs : array-like
            radii in grid units (same as the coordinates used by X in fit and
            predict). If not provided, defaults: 2000km, 1000km, 500km, 250km,
            125km, 62km, 31km, 15km. These defaults are scaled to coordinate
            units with to_meter
        to_meter : scalar
            Size of a grid cell, which is used to scale default radii from
            meters to units in coordinates.

        Notes
        -----
        Added in version 0.4.0

        Method from Glahn (2012) section 2 and section 11
        https://ams.confex.com/ams/92Annual/webprogram/Manuscript/Paper198533/
            AMS2012_preprint.pdf

        Basic gridding approach algorithm:

            A = first guess
            for r in rs:
                # i where d_i < r
                w_i = s_i * (r**2 - d_i**2) / (r**2 + d_i**2)
                w_i = w_i / sum(w_i)
                C = sum(w_i * (O_i - A))
                A += C

        Default radii:
            From  Djalalova (2015, 10.1016/j.atmosenv.2015.02.021)

        s_i is an addition to Glahn that allows for weighting of individual
            observations. This allows combining, for example, regulatory grade
            observations with low-cost sensors

        Basic smoothing approach algorith:

            r_x = min(d_i) for each x
            for x in X:
                # for all grid points where the distance from x (d_g) is less
                # than (r_x): d_g < r_x
                SA = (A_g / r_g) / sum(1/r_g)

        Several options should be considered for future development:

        * elevation dependence (aka VCE)
        * land/water differential smoothing
        * ridge awareness in smoothing
        """
        if rs is None:
            rs = np.array([2e3, 1e3, 5e2, 250, 125, 62, 31, 15]) * 1e3
            rs = rs / to_meter
        self._rs = rs

    def fit(self, X, y=None, s=None):
        """
        Arguments
        ---------
        X : array-like
            n by 2 array of coordinates that should be provided in grid units
        y : array-like
            n or n x m array of results
        s : array-like
            Scaling parameter for weights. This allows individual observations
            to be weighted based on uncertainty or some arbitrary value.

        Returns
        -------
        None
        """
        from sklearn.neighbors import NearestNeighbors

        self._nn = NearestNeighbors()
        _X = np.asarray(X)
        self._nn.fit(_X)
        chkX = self.get_fit_x()
        assert np.allclose(chkX, _X)
        self._y = np.asarray(y)
        if s is None:
            s = np.ones_like(self._y)
        self._s = np.asarray(s)
        assert self._y.shape[0] == _X.shape[0]

    def get_fit_x(self):
        if hasattr(self._nn, '_fit_X'):
            _X = self._nn._fit_X
        elif self._nn._tree is not None:
            _X = self._nn._tree.data
        else:
            raise ValueError('GMOS must be fit before calling get_fit_x')
        return np.asarray(_X)

    def predict(
        self, X, A=None, smooth=True, both=False, loo=False, verbose=0
    ):
        """
        Iteratively apply correction from apriori (A) based on observed (y)
        values using radii provided at initialization. If smooth, apply a
        simple smoothing that gives addition weight to pixels near monitors.
        If both, return the A and SA as a tuple.

        Arguments
        ---------
        X : array-like
            n by 2 array of coordinates that should be provided in grid units
        A : array-like
            a priori (aka guess) of the prediction (shape = n)
        smooth : bool
            If True (default), apply smoothing based on near by predictions.
            If False, return unsmoothed surface.
        both : bool
            If True and smooth is True, return unsmoothed and smoothed surface.
            If False (default), return only either unsmoothed or smoothed.
        loo : bool
            If True, remove the closest point from all calculations.
        verbose : int
            Level of verbosity

        Returns
        -------
        A, SA, or A or SA : array-like
        """
        # a priori guess
        if A is None:
            A = np.zeros(X.shape[0], dtype='f')
        else:
            A = np.asarray(A)

        # find all neighbors once
        r = self._rs[0]
        dists, idxs = self._nn.radius_neighbors(X, radius=r)
        ys = np.array([
            self._y[idx]
            for idx in idxs
        ], dtype=object)
        _ones = np.array([
            np.ones_like(len(idx), dtype=self._y.dtype)
            for idx in idxs
        ], dtype=object)
        ss = np.array([
            self._s[idx]
            for idx in idxs
        ], dtype=object)
        r2 = r**2
        d2 = dists**2
        allws = (r2 - d2) / (r2 + d2)
        if loo:
            # set weights to 0 distance equal to 0 (leave-one-out)
            allws = np.array([
                w * (dists[wi] != 0).astype('i') for wi, w in enumerate(allws)
            ], dtype=object)

        # Iterative bias correction
        for r in self._rs:
            if verbose > 0:
                print(r, flush=True)
            if r == self._rs[0]:
                ws = allws
            else:
                # zero-out weights from further distances.
                ws = np.array([
                    allws[di] * (dist < r).astype('i')
                    for di, dist in enumerate(dists)
                ], dtype='object')
            num = np.array([wv.sum() for wv in (ss * ws * (ys - A))])
            den = np.ma.masked_values([w.sum() for w in ws * ss], 0)
            C = num / den
            A += C.filled(0)

        if not smooth:
            return A

        # Apply smoothing
        NN = self._nn.kneighbors(X=X, n_neighbors=1)[0][:, 0]
        snn = NNA()
        snn.fit(X, A)
        sdists, sidxs = snn._nn.radius_neighbors(X=X, radius=NN)

        # This may be wrong... I'm not sure if NN should be the distance from
        # its own nearest monitor or the distance to the nearest monitor of
        # the local grid point
        sws = 1 / NN

        ys = np.array([snn._y[idx] for idx in sidxs], dtype=object)
        snum = np.array([wys.sum() for wys in (ys * sws)])
        sden = np.ma.masked_values([w.sum() for w in (sws * _ones)], 0)
        sval = snum / sden
        SA = sval
        # Special consideration of the 4 cells around the station
        # Glahn 2012 "With special consideration very close to the station, the
        # four grid points surrounding the station are left unchanged, while
        # the ones far away from the station may be considerably smoothed."
        # Currently assuming distances are in nominal grid units
        SA[NN < 0.5] = A[NN < 0.5]

        if both:
            return A, SA
        else:
            return SA
