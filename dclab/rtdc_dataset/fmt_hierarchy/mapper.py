import numpy as np


def map_indices_child2parent(child, child_indices):
    """Map child RTDCBase event indices to parent RTDCBase

    Given a hierarchy child and indices defined for that child,
    return the corresponding indices for its parent.

    For instance, a child is defined in such a way that it
    has every second event of its parent (`parent.filter.all[::2]=False`
    i.e. the filtering array is `[False, True, False, ...]`). When passing
    `child_indices=[2,3,4]`, the return value of this method would be
    `parent_indices=[5,7,9]` (indexing starts at 0)`. Index 5 in the
    parent dataset corresponds to index 2 in the child dataset.

    Parameters
    ----------
    child: RTDC_Hierarchy
        RTDCBase hierarchy child to map from
    child_indices: 1d int ndarray
        integer indices in `child`

    Returns
    -------
    parent_indices: 1d int ndarray
        integer indices in `child.hparent`
    """
    parent = child.hparent
    # filters
    pf = parent.filter.all
    # indices corresponding to all child events
    idx = np.where(pf)[0]  # True means present in the child
    # indices corresponding to selected child events
    parent_indices = idx[child_indices]
    return parent_indices


def map_indices_child2root(child, child_indices):
    """Map RTDC_Hierarchy event indices to root RTDCBase

    Like :func:`map_indices_child2parent`, but map the
    child indices to the root parent.

    Parameters
    ----------
    child: RTDC_Hierarchy
        RTDCBase hierarchy child to map from
    child_indices: 1d ndarray
        integer indices in `child`

    Returns
    -------
    root_indices: 1d ndarray
        integer indices in the child's root parent
        (not necessarily the indices of `child.hparent`)
    """
    while True:
        indices = map_indices_child2parent(child=child,
                                           child_indices=child_indices)
        if child.hparent.format == "hierarchy":
            child = child.hparent
            child_indices = indices
        else:
            break
    return indices


def map_indices_parent2child(child, parent_indices):
    """Map parent RTDCBase event indices to RTDC_Hierarchy child

    Given a hierarchy child and indices defined for its `child.hparent`,
    return the corresponding indices for the `child`.

    Parameters
    ----------
    child: RTDC_Hierarchy
        RTDCBase hierarchy child to map to
    parent_indices: 1d ndarray
        integer indices in `child.hparent`

    Returns
    -------
    child_indices: 1d ndarray
        integer indices in `child`, corresponding to `parent_indices`
        in `child.hparent`
    """
    parent = child.hparent
    # this boolean array defines `child` in the parent
    pf = parent.filter.all
    # all event indices in parent that define `child`
    pf_loc = np.where(pf)[0]
    # boolean array with size `len(child)` indicating where the
    # `parent_indices` are set.
    same = np.in1d(pf_loc, parent_indices)
    return np.where(same)[0]


def map_indices_root2child(child, root_indices):
    """Map root RTDCBase event indices to RTDC_Hierarchy child

    Like :func:`map_indices_parent2child`, but accepts the
    `root_indices` and map them to `child`.

    Parameters
    ----------
    child: RTDCBase
        RTDCBase hierarchy child to map to
    root_indices: 1d ndarray
        integer indices in the root parent of `child`

    Returns
    -------
    child_indices: 1d ndarray
        integer indices in `child`, corresponding to `root_indices`
        in `child`s root parent
    """
    # construct hierarchy tree containing only RTDC_Hierarchy instances
    hierarchy = [child]
    while True:
        if child.hparent.format == "hierarchy":
            # the parent is a hierarchy tree
            hierarchy.append(child.hparent)
            child = child.hparent
        else:
            break

    indices = root_indices
    for hp in hierarchy[::-1]:  # reverse order
        # For each hierarchy parent, map the indices down the
        # hierarchy tree.
        indices = map_indices_parent2child(child=hp,
                                           parent_indices=indices)
    return indices
