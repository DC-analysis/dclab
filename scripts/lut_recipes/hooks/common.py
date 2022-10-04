import numpy as np
from scipy.optimize import fsolve


def find_upper_intersection(func, points, xrange):
    options = []
    for ii in range(len(points)):
        x1, y1 = points[ii]
        x2, y2 = points[ii-1]
        def funcs(x): return y1 + (y2-y1)/(x2-x1)*(x-x1)
        ret = fsolve(lambda x: func(x) - funcs(x), xrange[0])
        for r in ret:
            if xrange[0] < r < xrange[1]:
                options.append([r, func(r)])
    options = np.array(options)
    # If there are multiple options, choose the one that is closest to
    # xrange[0].
    ido = np.argmin(np.abs(options[:, 0] - xrange[0]))
    return options[ido]
