__version__ = '0.3.3'
__all__ = ['NNA', '__version__']


from sklearn.base import MultiOutputMixin, RegressorMixin, BaseEstimator
import numpy as np


_def_maxweight = 1e20
_def_maxdist = np.inf


def get_vna_ridge_lengths(xy):
    """convenience function for Laplace Weights"""
    from scipy.spatial import Voronoi
    vd = Voronoi(xy)
    isme = (vd.ridge_points == (vd.points.shape[0] - 1))
    hasme = isme.any(1)
    isother = hasme[:, None] & ~isme
    ridge_xy = vd.vertices[np.array(vd.ridge_vertices)[np.where(hasme)[0]]]
    ridge_lengths = (np.diff(ridge_xy, axis=1)**2).sum(-1)[:, 0]**.5
    ridge_other = vd.ridge_points[isother]
    ridge_idx = np.argsort(ridge_other)
    ridge_other = ridge_other[ridge_idx]
    ridge_lengths = ridge_lengths[ridge_idx]
    return ridge_other, ridge_lengths


class NNA(BaseEstimator, MultiOutputMixin, RegressorMixin):
    def __init__(
        self, k=10, power=-2, method='nearest', maxweight=_def_maxweight,
        maxdist=_def_maxdist, loo=False, verbose=0
    ):
        """
        Nearest Neighbor Averaging (NNA) object is designed to support 2D
        neighbor based interpolation. It is designed in the scikit-learn style,
        with fit and predict methods. It is not part of the scikit-learn
        package. Perhaps one day. Currently, supports:

        * Several Neighbor Selection Methods
          * Nearest Neighbors (k-nearest)
          * Voronoi neighbors from k-nearest
        * Alternative Neighbor Weights
          * distance power weightings (e.g., d**-2)
          * Laplace weightings for Voronoi
        * Leave one out (loo) prediction for simple cross validation

        Basic Usage:

            nn = NNA()
            nn.fit(X, z)
            zhat = nn.predict(X)

        The predict method has keywords that support multiple neighbor
        selection methods and weights. See help(NNA.predict) for more details.
        An example below, creates synthetic data, fits and predicts with three
        alternative methods and prints the RMSE for each.

        Example Usage:

            import numpy as np
            import nna_methods

            # create synthetic data
            n = 90
            x = np.arange(n, dtype='f')
            X, Y = np.meshgrid(x, x)
            Z = X * 5 + Y * 10 + 20
            XY = np.array([X.ravel(), Y.ravel()]).T

            # Random subsample of space
            xy = np.array([
                np.random.randint(n, size=n),
                np.random.randint(n, size=n)
            ]).T
            z = Z[xy[:, 0], xy[:, 1]]

            # Fit xy to z once
            nn = nna_methods.NNA()
            nn.fit(xy + 1e-6, z)

            # Reconstruct image using different methods
            for method in ['nearest', 'voronoi', 'laplace']:
              Zhat = nn.predict(XY, method=method, k=30).reshape(Z.shape)
              rmse = ((np.ma.masked_invalid(Zhat) - Z)**2).mean()**.5
              print(method, rmse)

        All keyword arguments are used as defaults in predict.

        Arguments
        ---------
        X : array-like
            Target coordinates n x 2
        k : int
            number of nearest neighbors
        power : scalar
            distance power (default -2) or None to return distances
        maxweight : float
            maximum weight, which prevents np.inf at fit locations.
        maxdist : int
            max distance to be used in weights. Values above maxdist will be
            masked.
        method : str
            Choices are nearest, voronoi, laplace:
              * nearest : Nearest n neighbors with IDW weights
              * voronoi : Voronoi neighbors (within n) with IDW weights
              * laplace : Voronoi neighbors (within n) with laplacian weights
                          shared polygon face length divided by distance of
                          centroids.
        loo : bool
            If True, leave out the nearest neighbor. Good for validation.
        verbose : int
            Level of verbosity

        Note:
            mindist has been replaced with maxweight.
        """
        self.k = k
        self.power = power
        self.method = method
        self.maxweight = maxweight
        self.maxdist = maxdist
        self.loo = loo
        self.verbose = verbose

    def fit(self, X, y=None):
        """
        Arguments
        ---------
        X : array-like
            n by 2 array of coordinates
        y : array-like
            n or n x m array of results

        Returns
        -------
        None
        """
        from sklearn.neighbors import NearestNeighbors

        self._nn = NearestNeighbors()
        _X = np.asarray(X)
        self._nn.fit(_X)
        assert np.allclose(np.asarray(self._nn._tree.data), _X)
        self._y = np.asarray(y)
        assert self._y.shape[0] == _X.shape[0]

    def nn(self, X, k=10, sort=True):
        """
        Arguments
        ---------
        X : array-like
            n by 2 array of coordinates
        k : int
            number of nearest neighbors
        sort : bool
            Sort neighbor distances and idx by distance

        Returns
        -------
        distn : array-like
            k nearest distances
        idxn : array-like
            the indices of the k nearest neighbors in df
        """
        dist, idx = self._nn.kneighbors(X, k)

        if sort:
            distidx = np.argsort(dist, axis=1)
            dist = np.take_along_axis(dist, distidx, axis=1)
            idx = np.take_along_axis(idx, distidx, axis=1)

        return dist, idx

    def idw_nn_wgts(
        self, X, k=10, power=-2, maxweight=_def_maxweight, maxdist=None,
        loo=False
    ):
        """
        Calculate nearest neighbor weights

        Arguments
        ---------
        X : array-like
            Target coordinates n x 2
        k : int
            number of nearest neighbors
        power : scalar
            distance power (default -2) or None to return distances
        maxweight : float
            maximum weight, which prevents np.inf at fit locations.
        maxdist : int
            max distance to be used in weights. Values above maxdist will be
            masked.
        loo : bool
            If True, leave out the nearest neighbor. Good for validation.

        Returns
        -------
        wgt : array-like
            distn**power
        idxn :
            index for weights
        """
        if maxdist is None:
            maxdist = np.inf
        if loo:
            k = k + 1
        dist, idx = self.nn(X, k=k)
        if loo:
            dist = dist[:, 1:]
            idx = idx[:, 1:]

        dist = np.ma.masked_greater(dist, maxdist)

        if power is None:
            wgt = dist
        else:
            mindist = maxweight**(1/power)
            wgt = np.minimum(maxweight, np.maximum(mindist, dist)**power)
            wgt = wgt / wgt.sum(1)[:, None]
        return wgt, idx

    def idw_vn_wgts(
        self, X, k=30, power=-2, maxweight=_def_maxweight, maxdist=None,
        loo=False
    ):
        """
        Calculate voronoi neighbor weights. Same as idw_nn_wgts, but masks
        values from non-voronoi neighbors.

        Arguments
        ---------
        X : array-like
            Target coordinates n x 2
        k : int
            number of nearest neighbors
        power : scalar
            distance power (default -2) or None to return distances
        maxweight : float
            maximum weight, which prevents np.inf at fit locations.
        maxdist : int
            max distance to be used in weights. Values above maxdist will be
            masked.
        loo : bool
            If True, leave out the nearest neighbor. Good for validation.

        Returns
        -------
        wgt : array-like
            distn**power
        idxn :
            index for weights
        """
        if maxdist is None:
            maxdist = np.inf
        if loo:
            k = k + 1
        dist, idx = self.nn(X, k=k)
        if loo:
            dist = dist[:, 1:]
            idx = idx[:, 1:]

        isvn = self.findvn(X, idx)
        dist = np.ma.masked_where(~isvn, dist)
        if power is None:
            wgt = dist
        else:
            mindist = maxweight**(1/power)
            wgt = np.ma.masked_greater(
                np.maximum(mindist, dist),
                maxdist
            )**power
            wgt = np.minimum(maxweight, wgt)
            wgt = wgt / wgt.sum(1)[:, None]

        return wgt, idx

    def laplace_vn_wgts(
        self, X, k=10, power=-2, maxweight=_def_maxweight, maxdist=None,
        loo=False
    ):
        """
        Calculate nearest neighbor weights

        Arguments
        ---------
        X : array-like
            Target coordinates n x 2
        k : int
            number of nearest neighbors
        power : scalar
            distance power (default -2) or None to return distances.
            Currently unused, but accepted for compatibility with idw_nn_wgts,
            and idw_vn_wgts
        maxweight : float
            maximum weight, which prevents np.inf at fit locations.
        maxdist : int
            max distance to be used in weights. Values above maxdist will be
            masked.
        loo : bool
            If True, leave out the nearest neighbor. Good for validation.

        Returns
        -------
        wgt : array-like
            ridge length / distance
        idxn :
            index for weights
        """
        if maxdist is None:
            maxdist = np.inf
        if loo:
            k = k + 1
        dist, idx = self.nn(X, k=k)
        if loo:
            dist = dist[:, 1:]
            idx = idx[:, 1:]

        X = np.asarray(X)
        isvn = self.findvn(X, idx)
        _X = np.asarray(self._nn._tree.data)
        rls = np.zeros_like(dist)
        n = idx.shape[0]
        if self.verbose < 1:
            chkn = n + 1
        else:
            chkn = n // 200 + 1
        for i, iidx in enumerate(idx):
            if i > 0 and (i % chkn) == 0:
                print(f'\r{i/n:.1%}', end='', flush=True)
            vnidx = iidx[isvn[i]]
            vnxy = np.concatenate([_X[vnidx], X[i][None, :]], axis=0)
            ri, rl = get_vna_ridge_lengths(vnxy)
            rli = np.zeros(isvn[i].sum(), dtype='d')
            rli[ri] = rl
            rls[i, isvn[i]] = rli

        mindist = np.maximum(rls / maxweight, 1e-20)
        laplace_wgt = rls / np.maximum(mindist, dist)
        laplace_wgt = np.minimum(maxweight, laplace_wgt)
        laplace_wgt = laplace_wgt / laplace_wgt.sum(1)[:, None]
        return laplace_wgt, idx

    def findvn(self, X, idxn):
        """
        Coupled with results from idw_nn_wgts, find Voronoi Neigbors in
        nearest neighbors.

        Arguments
        ---------
        X: grid file (ie target)
        idxn: nearest neighbors for each

        Returns
        -------
        isvna : for each hasnear, and each k neighbor, is it a Voronoi Neighbor
        """
        from scipy.spatial import Delaunay

        k = idxn.shape[1]
        n = X.shape[0]

        isvn = np.zeros((n, k), dtype='bool')
        didx = np.arange(k)
        if self.verbose < 1:
            chk = n + 1
        else:
            chk = n // 200 + 1

        # Get locations with target as last point
        xy = np.asarray(self._nn._tree.data)[idxn]
        vnxy = np.concatenate([xy, np.asarray(X)[:, None, :]], axis=1)

        # For each target with near points, calc Delaunay and find neighbors
        # Tag neighbors as is Voronoi Neighbor isvn
        for i in range(n):
            if i > 0 and (i % chk) == 0:
                print(f'{i / n:.1%}', end='\r')
            newxy = vnxy[i]
            tric = Delaunay(newxy)
            tri_indicies, tri_neighbors = tric.vertex_neighbor_vertices
            cidx = tri_neighbors[tri_indicies[k]:tri_indicies[k + 1]]
            isvn[i] = np.in1d(didx, cidx)

        # if there are not neighbors, then you are in the same cell
        # In that case, closest three should be used have 100%
        isvn[np.where(~isvn.any(1))[0], :3] = True

        return isvn

    def predict(
        self, X, k=None, power=None, maxweight=None, maxdist=None, method=None,
        loo=None
    ):
        """
        Keyword arguments can be used to supersede keywords used to initialize
        the object. The fit command is independent, so results with superseding
        keywords are the same as if the object had been initialized and fit
        with those arguments.

        Arguments
        ---------
        X : array-like
            Target coordinates n x 2
        k : int
            number of nearest neighbors
        power : scalar
            distance power (default -2) or None to return distances
        maxweight : float
            maximum weight, which prevents np.inf at fit locations.
        maxdist : int
            max distance to be used in weights. Values above maxdist will be
            masked.
        method : str
            Choices are nearest, voronoi, laplace:
              * nearest : Nearest n neighbors with IDW weights
              * voronoi : Voronoi neighbors (within n) with IDW weights
              * laplace : Voronoi neighbors (within n) with laplacian weights
                          shared polygon face length divided by distance of
                          centroids.
        loo : bool
            If True, leave out the nearest neighbor. Good for validation.


        Returns
        -------
        yhat : array-like
            array of predictions (yhat). If y was 1-d, then array has shape
            n = (n=X.shape[0]). If y was 2-d, then array has the n x m.
        """
        # Use defaults from initialization
        if k is None:
            k = self.k
        if power is None:
            power = self.power
        if maxweight is None:
            maxweight = self.maxweight
        if maxdist is None:
            maxdist = self.maxdist
        if method is None:
            method = self.method
        if loo is None:
            loo = self.loo

        if method == 'voronoi':
            wgtf = self.idw_vn_wgts
        elif method == 'laplace':
            wgtf = self.laplace_vn_wgts
        elif method == 'nearest':
            wgtf = self.idw_nn_wgts
        else:
            raise KeyError(f'method {method} unknown; use mearest or voronoi')
        wgt, idx = wgtf(
            X, k=k, power=power, maxweight=maxweight, maxdist=maxdist, loo=loo
        )
        if self._y.ndim == 1:
            yhat = (self._y[idx] * wgt).sum(1)
            return yhat
        else:
            yhats = []
            for y in self._y.T:
                yhat = (y[idx] * wgt).sum(1)
                yhats.append(yhat)
            return np.array(yhats).T

    def cross_validate(self, X, y, df=None, ykey='y', **kwds):
        """
        Use nn to perform a KFold cross validation where kwds are passed to
        KFold.

        Arguments
        ---------
        X : array-like
            n by 2 array of coordinates
        y : array-like
            n or n x m array of results
        df : pandas.DataFrame or None
            If None, create a new dataframe with cross-validation results.
            Otherwise, add cross-validation results to df
        ykey : str
            Name of value being predicted.
        **kwds : mappable
            Passed to sklearn.model_selection.KFold to control cross-validation
            options.

        Returns
        -------
        df : pandas.DataFrame
            Returns a dataframe with CV_ykey and CV_ykey_fold
            where CV_ykey is the cross-validation predictions and CV_ykey_fold
            will be the fold-part that the prediction was made
        """
        from sklearn.model_selection import KFold
        import pandas as pd

        kwds.setdefault('n_splits', 10)
        kwds.setdefault('shuffle', True)
        kwds.setdefault('random_state', 1)

        X = np.asarray(X)
        y = np.asarray(y)
        kf = KFold(**kwds)
        zhats = np.zeros_like(y) * np.nan
        fold = np.zeros_like(y, dtype='i') * np.nan
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            self.fit(X[train_index], y[train_index])
            zhats[test_index] = self.predict(X[test_index])
            fold[test_index] = i

        if df is not None:
            df[f'CV_{ykey}'] = zhats
            df[f'CV_{ykey}_fold'] = fold
        else:
            df = pd.DataFrame({f'CV_{ykey}': zhats, f'CV_{ykey}_fold': fold})

        df.attrs.update(kwds)

        return df


if __name__ == '__main__':
    import argparse
    import os
    import pandas as pd

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.description = """
Applies nearest neighbor interpolation using nna_methods.NNA object.
This has 3 main steps:

1. Read source csv with xkey, ykey and zkey
2. Fit NNA object predict zkey using xkey and ykey
3. Read target csv (defaults to source csv for Leave-One-Out assessment.
4. Predict target locations (xkey, ykey)
5. Save out as a CSV

Step 4 uses options to determine the number of neighbors, the max
distance for neighbors, the minimum distance to use in weighting
calculations, the power to use with distance for weights, and the
method of neighbor selection/weighting.
"""
    parser.epilog = """
For example, daily.csv has Longitude, Latitude, and Sample Measurement
fields -- you can run a leave-one-out evaluation with the command below.

$ python nna_neighbors.py --xkey=Longitude --ykey=Latitude \
 --zkey="Sample Measurement" \
 daily.csv nna_daily.csv

If you wanted to predict unknown locations, simply add a csv path for the
target locations. In this case, target.csv must have Longitude and Latitude.


$ python nna_neighbors.py --xkey=Longitude --ykey=Latitude \
 --zkey="Sample Measurement" \
 daily.csv nna_daily.csv target.csv
"""
    parser.add_argument(
        '--cross-validation', default=False, action='store_true',
        help='Perform a 10-fold cross validation'
    )
    parser.add_argument('--neighbors', default=10)
    parser.add_argument('--power', default=-2, type=float)
    parser.add_argument('--maxweight', default=_def_maxweight, type=float)
    parser.add_argument('--maxdist', default=None, type=float)
    parser.add_argument(
        '--method', default='nearest',
        choices={'nearest', 'voronoi', 'laplace'}
    )
    parser.add_argument('--loo', default=False, action='store_true')
    parser.add_argument('--xkey', default='x')
    parser.add_argument('--ykey', default='y')
    parser.add_argument('--zkey', default='z')
    parser.add_argument('inputcsv')
    parser.add_argument('outputcsv')
    parser.add_argument('targetcsv', nargs='?')

    args = parser.parse_args()

    if os.path.exists(args.outputcsv):
        raise IOError(f'{args.outputcsv} exists; delete to remake')

    srcdf = pd.read_csv(args.inputcsv)
    srcdf['X'] = srcdf[args.xkey]
    srcdf['Y'] = srcdf[args.ykey]
    srcdf['Z'] = srcdf[args.zkey]

    if args.targetcsv is None:
        tgtdf = srcdf[['X', 'Y']].copy()
    else:
        tgtdf = pd.read_csv(args.targetcsv)
        tgtdf['X'] = tgtdf[args.xkey]
        tgtdf['Y'] = tgtdf[args.ykey]

    opts = dict(
        maxweight=args.maxweight, maxdist=args.maxdist, loo=args.loo,
        method=args.method, k=args.neighbors, power=args.power
    )

    nn = NNA(**opts)

    if args.cross_validation:
        tgtdf = srcdf.copy()
        nn.cross_validate(srcdf[['X', 'Y']], srcdf['Z'], df=tgtdf)
    else:
        nn.fit(srcdf[['X', 'Y']], srcdf['Z'])
        tgtdf['Zhat'] = nn.predict(tgtdf[['X', 'Y']], **opts)
        if args.targetcsv is None:
            tgtdf['Z'] = srcdf['Z']

    tgtdf.to_csv(args.outputcsv, index=False)
