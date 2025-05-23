__version__ = '0.7.0'
__all__ = ['NNA', 'GMOS', 'DelaunayInterp', 'NNABlender', '__version__']


from ._nna import NNA
from ._delaunay import DelaunayInterp
from ._nna import _def_maxweight
from ._gmos import GMOS
from ._blend import NNABlender

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
