from sklearn.base import MultiOutputMixin, RegressorMixin, BaseEstimator
import numpy as np


class DelaunayInterp(BaseEstimator, MultiOutputMixin, RegressorMixin):
    def __init__(self, *args, power=-2, **kwds):
        """
        DelaunayInterp interpolates values from the three vertices of each
        Delaunay simplex within that simplex.

        Interpolated values are set following:
            y(n, m) = sum(w_i * y_{i,m})
            where i is 1 thru 3
            w'_i = d_i**power
            w_i = w_i / sum(w'_i)

        Example:

            import matplotlib.pyplot as plt
            from nna_methods import DelaunayInterp
            import numpy as np

            x = np.array([
                np.random.uniform(-np.pi, np.pi, size=100),
                np.random.uniform(-np.pi, np.pi, size=100)
            ]).T
            y = (
                np.cos(x[:, 0]) + np.cos(x[:, 1])
                + np.random.normal(0, .2, size=100)
            )
            g = np.linspace(-np.pi, np.pi)
            X = np.array(np.meshgrid(g, g)).T.reshape(-1, 2)
            nn = DelaunayInterp()
            nn.fit(x, y)
            yhat = nn.predict(X, power=-2)
            fig, ax = plt.subplots()
            Z = yhat.reshape(g.size, g.size)
            qm = ax.pcolormesh(g, g, Z, cmap='nipy_spectral')
            fig.colorbar(qm)
            ax.scatter(
                x[:, 1], x[:, 0], c=y, norm=qm.norm, cmap=qm.cmap,
                edgecolors='white'
            )
        """
        self._power = power

    def get_fit_x(self):
        """
        """
        return self._fitX

    def fit(self, X, y=None):
        """
        Arguments
        ---------
        X : array-like
            Array shaped (n, d) where d is usually 2 (x, y).
        y : array-like
            Array shaped (n, y) where y is the number of y vectors to
            interpolate

        Returns
        -------
        None
        """
        self._fitX = np.asarray(X)
        self._tric = None
        self._fity = np.asarray(y)

    def predict(self, X, power=None):
        """
        Arguments
        ---------
        X : array-like
            Array shaped (n, d) where d is usually 2 (x, y).
        y : array-like
            Array shaped (n, m) where m is the number of y vectors to
            interpolate

        Returns
        -------
        Y : array-like
            Array shaped (n, m) where m is the number of y vectors to
            interpolate and values are set to y(n, m) = sum(w_i * y_{i,m})
            where i is 1 thru 3 and w'_i = d_i**power and w_i = w_i / sum(w'_i)
        """
        from scipy.spatial import Delaunay
        if self._tric is None:
            self._tric = Delaunay(self._fitX)

        if power is None:
            power = self._power

        X = np.asarray(X)
        tric = self._tric
        si = tric.find_simplex(X)
        vX = X[si > -1]
        fitX = self.get_fit_x()
        nidx = tric.simplices[si[si > -1]].ravel()
        nx = fitX[nidx, 0].reshape(-1, 3)
        ny = fitX[nidx, 1].reshape(-1, 3)
        dx = vX[:, 0, None] - nx
        dy = vX[:, 1, None] - ny
        d = (dx**2 + dy**2)**.5
        wgt = d**power
        wgt = wgt / wgt.sum(1, keepdims=True)
        fity = self._fity[nidx]
        if fity.ndim == 1:
            fity = fity[:, None]

        yshape = (X.shape[0], fity.shape[-1])
        Y = np.ones(yshape, dtype=self._fity.dtype) * np.nan
        for yi, y in enumerate(fity.T):
            yhat = (wgt * y.reshape(-1, 3)).sum(-1)
            Y[si > -1, yi] = yhat

        return Y.squeeze()
