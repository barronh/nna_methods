def gettest(f=None):
    """
    Get a data frame with X, Y, and Z where X and Y range from -1, 1 and Z
    is defined by f.

    Arguments
    ---------
    f : function or None
        If f is None, a simple Z = cos(x * pi) + cos(y * pi) + 4

    Returns
    -------
    df : pandas.DataFrame
        With X, Y, and Z and a random 10% marked for use in fitting (forfit)
        where xyi = np.arange(X.size); numpy.random.seed(0); np.shuffle(xyi)
        forfit[xyi[:1000]] = True
    """
    import pandas as pd
    import numpy as np
    if f is None:
        def f(x, y):
            return np.cos(x * np.pi) + np.cos(y * np.pi) + 4
    np.random.seed(0)
    xyi = np.arange(100*100)
    np.random.shuffle(xyi)
    x = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, x)
    useme = np.zeros(X.shape, dtype='bool')
    useme.ravel()[xyi[:1000]] = True
    df = pd.DataFrame(dict(
        X=X.ravel(),
        Y=Y.ravel(),
        forfit=useme.ravel()
    ))
    df['Z'] = f(df['X'], df['Y'])

    return df


def test_nn_loo(f=None):
    """
    Simple testing function for all nearest neighbor averaging approaches:
    nearest, voronoi and laplace all with power=-2 and k=30 run using leave
    one out (loo=True).

    Arguments
    ---------
    f : function to be passed to gettest

    Returns
    -------
    df : pandas.DataFrame
        with Zhat_nn, Zhat_vn, Zhat_ln, Zhat_nn2 (from predict with 2 Y
        variables) and Zhalf_nn3 (the second predicted variable 1/2 * Z).
    """
    from .. import NNA

    df = gettest(f=f)
    fitdf = df.query('forfit == True').copy()
    nn = NNA()
    nn.fit(fitdf[['X', 'Y']], fitdf['Z'])
    df['Zhat_nn'] = nn.predict(
        df[['X', 'Y']], loo=True, method='nearest', k=30
    )
    df['Zhat_vn'] = nn.predict(
        df[['X', 'Y']], loo=True, method='voronoi', k=30
    )
    df['Zhat_ln'] = nn.predict(
        df[['X', 'Y']], loo=True, method='laplace', k=30
    )
    fitdf['ZHALF'] = fitdf['Z'] * 0.5
    nn.fit(fitdf[['X', 'Y']], fitdf[['Z', 'ZHALF']])
    zs = nn.predict(
        df[['X', 'Y']], loo=True, method='nearest', k=30
    )
    df['Zhat_nn2'] = zs[:, 0]
    df['Zhalf_nn3'] = zs[:, 1]
    return df


def test_cross_validate(f=None):
    """
    Basic test of the cross_validate method with voronoi and nearest

    Arguments
    ---------
    f : function to be passed to gettest

    Returns
    -------
    df : pandas.DataFrame
        with CV_VNA and CV_NNA cross-validation predictions.
    """
    from .. import NNA

    df = gettest(f=f)
    nn = NNA(method='voronoi', k=30)
    nn.cross_validate(df[['X', 'Y']], df['Z'], df=df, ykey='VNA')
    nn = NNA(method='nearest', k=10)
    nn.cross_validate(df[['X', 'Y']], df['Z'], df=df, ykey='NNA')
    return df
