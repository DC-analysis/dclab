"""Content-based downsampling of ndarrays"""
import numpy as np
cimport numpy as cnp

from .cached import Cache

# We are using the numpy array API
cnp.import_array()
ctypedef cnp.uint8_t uint8
ctypedef cnp.uint32_t uint32
ctypedef cnp.int64_t int64

def downsample_rand(a, samples, remove_invalid=False, ret_idx=False):
    """Downsampling by randomly removing points

    Parameters
    ----------
    a: 1d ndarray
        The input array to downsample
    samples: int
        The desired number of samples
    remove_invalid: bool
        Remove nan and inf values before downsampling
    ret_idx: bool
        Also return a boolean array that corresponds to the
        downsampled indices in `a`.

    Returns
    -------
    dsa: 1d ndarray of size `samples`
        The pseudo-randomly downsampled array `a`
    idx: 1d boolean array with same shape as `a`
        Only returned if `ret_idx` is True.
        A boolean array such that `a[idx] == dsa`
    """
    # fixed random state for this method
    rs = np.random.RandomState(seed=47).get_state()
    np.random.set_state(rs)

    cdef uint32 samples_int = np.uint32(samples)

    if remove_invalid:
        # slice out nans and infs
        bad = np.isnan(a) | np.isinf(a)
        pool = a[~bad]
    else:
        pool = a

    if samples_int and (samples_int < pool.shape[0]):
        keep = np.zeros_like(pool, dtype=bool)
        keep_ids = np.random.choice(np.arange(pool.size),
                                    size=samples_int,
                                    replace=False)
        keep[keep_ids] = True
        dsa = pool[keep]
    else:
        keep = np.ones_like(pool, dtype=bool)
        dsa = pool

    if remove_invalid:
        # translate the kept values back to the original array
        idx = np.zeros(a.size, dtype=bool)
        idx[~bad] = keep
    else:
        idx = keep

    if ret_idx:
        return dsa, idx
    else:
        return dsa


@Cache
def downsample_grid(a, b, samples, remove_invalid=False, ret_idx=False):
    """Content-based downsampling for faster visualization

    The arrays `a` and `b` make up a 2D scatter plot with high
    and low density values. This method takes out points at
    indices with high density.

    Parameters
    ----------
    a, b: 1d ndarrays
        The input arrays to downsample
    samples: int
        The desired number of samples
    remove_invalid: bool
        Remove nan and inf values before downsampling; if set to
        `True`, the actual number of samples returned might be
        smaller than `samples` due to infinite or nan values.
    ret_idx: bool
        Also return a boolean array that corresponds to the
        downsampled indices in `a` and `b`.

    Returns
    -------
    dsa, dsb: 1d ndarrays of shape (samples,)
        The arrays `a` and `b` downsampled by evenly selecting
        points and pseudo-randomly adding or removing points
        to match `samples`.
    idx: 1d boolean array with same shape as `a`
        Only returned if `ret_idx` is True.
        A boolean array such that `a[idx] == dsa`
    """
    cdef uint32 samples_int = np.uint32(samples)

    # fixed random state for this method
    rs = np.random.RandomState(seed=47).get_state()

    keep = np.ones_like(a, dtype=bool)
    bad = np.isnan(a) | np.isinf(a) | np.isnan(b) | np.isinf(b)
    good = ~bad

    # We are generally not interested in bad data.
    keep[bad] = False
    bd = b[good]
    ad = a[good]

    cdef int64 diff

    if samples_int and samples_int < ad.size:
        # 1. Produce evenly distributed samples
        # Choosing grid-size:
        # - large numbers tend to show actual structures of the sample,
        #   which is not desired for plotting
        # - small numbers tend will not result in too few samples and,
        #   in order to reach the desired samples, the data must be
        #   upsampled again.
        # 300 is about the size of the plot in marker sizes and yields
        # good results.
        grid_size = 300
        # The events on the grid to process
        toproc = np.ones((grid_size, grid_size), dtype=np.uint8)

        x_discrete = np.array(norm(ad) * (grid_size - 1), dtype=np.uint32)
        y_discrete = np.array(norm(bd) * (grid_size - 1), dtype=np.uint32)

        # The events to keep
        keepd = np.zeros_like(ad, dtype=np.uint8)
        populate_grid(x_discrete=x_discrete,
                      y_discrete=y_discrete,
                      toproc=toproc,
                      keepd=keepd
                      )

        keepdb = np.array(keepd, dtype=bool)

        # 2. Make sure that we reach `samples` by adding or
        # removing events.
        diff = np.sum(keepdb) - samples_int
        if diff > 0:
            # Too many samples
            rem_indices = np.where(keepdb)[0]
            np.random.set_state(rs)
            rem = np.random.choice(rem_indices,
                                   size=diff,
                                   replace=False)
            keepdb[rem] = False
        elif diff < 0:
            # Not enough samples
            add_indices = np.where(~keepdb)[0]
            np.random.set_state(rs)
            add = np.random.choice(add_indices,
                                   size=abs(diff),
                                   replace=False)
            keepdb[add] = True

        # paulmueller 2024-01-03
        # assert np.sum(keepdb) <= samples_int, "sanity check"

        keep[good] = keepdb

        # paulmueller 2024-01-03
        # assert np.sum(keep) <= samples_int, "sanity check"

    if not remove_invalid:
        diff_bad = (samples_int or keep.size) - np.sum(keep)
        if diff_bad > 0:
            # Add a few of the invalid values so that in the end
            # we have the desired array size.
            add_indices_bad = np.where(bad)[0]
            np.random.set_state(rs)
            add_bad = np.random.choice(add_indices_bad,
                                       size=diff_bad,
                                       replace=False)
            keep[add_bad] = True

    # paulmueller 2024-01-03
    # if samples_int and not remove_invalid:
    #     assert np.sum(keep) == samples_int, "sanity check"

    asd = a[keep]
    bsd = b[keep]

    if ret_idx:
        return asd, bsd, keep
    else:
        return asd, bsd


def populate_grid(x_discrete, y_discrete, keepd, toproc):
    # Py_ssize_t is the proper C type for Python array indices.
    cdef int iter_size = x_discrete.size
    cdef Py_ssize_t ii
    cdef uint32[:] x_view = x_discrete
    cdef uint32[:] y_view = y_discrete
    # Numpy uses uint8 internally to represent boolean arrays
    cdef uint8[:] keepd_view = keepd
    cdef uint8[:, :] toproc_view = toproc

    for ii in range(iter_size):
        # filter for overlapping events
        xi = x_view[ii]
        yi = y_view[ii]
        if toproc_view[xi, yi]:
            toproc_view[xi, yi] = 0
            # include event
            keepd_view[ii] = 1


def valid(a, b):
    """Check whether `a` and `b` are not inf or nan"""
    return ~(np.isnan(a) | np.isinf(a) | np.isnan(b) | np.isinf(b))


def norm(a):
    """Normalize `a` with its min/max values"""
    rmin = a.min()
    rptp = a.max() - rmin
    return (a - rmin) / rptp
