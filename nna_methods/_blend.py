from ._nna import NNA


class NNABlender(NNA):
    def __init__(self, nnas, weights):
        """
        Calculate simultaneously blended NNA. Create weights for each nna and
        then normalize all weights to create a prediction.
        """
        for nn in nnas:
            if getattr(nn, '_y', None) is None:
                msg = 'All nnas must be fit before provided to NNABlender'
                raise ValueError(msg)
        self._nnas = nnas
        self._weights = weights

    def predict(
        self, X, k=None, power=None, maxweight=None, maxdist=None, method=None,
        loo=None, njobs=None
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
        njobs : int or None
            If None, process as serial operation.
            If int, use joblib.Parallel and joblib.delayed to run njobs
            parallel processes and concatenate results

        Returns
        -------
        yhat : array-like
            array of predictions (yhat). If y was 1-d, then array has shape
            n = (n=X.shape[0]). If y was 2-d, then array has the n x m.
        """
        import numpy as np
        if njobs is not None:
            from joblib import Parallel, delayed
            n = X.shape[0]
            ns = [n // njobs] * njobs
            ns[-1] += (n - sum(ns))
            print('Cells per job', ns)
            se = np.cumsum([0] + ns)
            with Parallel(n_jobs=njobs, verbose=10) as par:
                processed_list = par(
                    delayed(self.predict)(
                        X[s:e], k=k, power=power, maxweight=maxweight,
                        maxdist=maxdist, method=method, loo=loo, njobs=None
                    )
                    for s, e in zip(se[:-1], se[1:])
                )
            yout = np.ma.concatenate(processed_list, axis=0)
            return yout

        # Use defaults from initialization
        wgts = []
        ys = []
        for ni, nn in enumerate(self._nnas):
            nnmethod = method or nn.method
            nnpower = power or nn.power
            nnmaxweight = maxweight or nn.maxweight
            nnmaxdist = maxdist or nn.maxdist
            nnloo = loo or nn.loo
            nnloo = loo or nn.loo
            if nnmethod == 'voronoi':
                wgtf = nn.idw_vn_wgts
            elif nnmethod == 'nearest':
                wgtf = nn.idw_nn_wgts
            else:
                msg = f'method {method} unknown; use mearest or voronoi'
                raise KeyError(msg)

            dist, idx = wgtf(X, power=None, loo=nnloo)
            dist = np.ma.masked_greater(dist, nnmaxdist)
            wgt = self._weights[ni] * dist**nnpower
            wgt[:] = np.minimum(nnmaxweight, wgt)
            wgts.append(wgt)
            ys.append(nn._y[idx])

        wgt = np.concatenate(wgts, axis=1)
        wgt /= wgt.sum(1)[:, None]
        ys = np.concatenate(ys, axis=1)
        if nn._y.ndim == 1:
            yhat = (ys * wgt).sum(1)
            return yhat
        else:
            yhats = []
            for y in ys.T:
                yhat = (y[idx] * wgt).sum(1)
                yhats.append(yhat)
            return np.array(yhats).T
