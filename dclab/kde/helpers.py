import numpy as np


def get_bad_vals(x, y):
    return np.isnan(x) | np.isinf(x) | np.isnan(y) | np.isinf(y)


def ignore_nan_inf(kde_method):
    """Decorator that computes the KDE only for valid values

    Invalid positions in the resulting density are set to nan.
    """

    def kde_wrapper(events_x, events_y, xout=None, yout=None, *args, **kwargs):
        bad_in = get_bad_vals(events_x, events_y)
        if xout is None or yout is None:
            density = np.zeros_like(events_x, dtype=np.float64)
            bad_out = bad_in
            xo = yo = None
        else:
            density = np.zeros_like(xout, dtype=np.float64)
            bad_out = get_bad_vals(xout, yout)
            xo = xout[~bad_out]
            yo = yout[~bad_out]
        # Filter events
        ev_x = events_x[~bad_in]
        ev_y = events_y[~bad_in]
        if ev_x.size:
            density[~bad_out] = kde_method(ev_x, ev_y, xo, yo, *args, **kwargs)
        density[bad_out] = np.nan
        return density

    doc_add = (
        "\n    Notes\n"
        + "    -----\n"
        + "    This is a wrapped version that ignores nan and inf values."
    )
    kde_wrapper.__doc__ = kde_method.__doc__ + doc_add

    return kde_wrapper
