import numpy as np


def map_indices_child2parent(child, child_indices):
    """Map child RTDCBase event indices to parent RTDCBase

    Parameters
    ----------
    child: RTDC_Hierarchy
        hierarchy child with `child_indices`
    child_indices: 1d ndarray
        child indices to map

    Returns
    -------
    parent_indices: 1d ndarray
        hierarchy parent indices
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

    Parameters
    ----------
    child: RTDC_Hierarchy
        RTDCBase hierarchy child
    child_indices: 1d ndarray
        child indices to map

    Returns
    -------
    root_indices: 1d ndarray
        hierarchy root indices
        (not necessarily the indices of `parent`)
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

    Parameters
    ----------
    child: RTDC_Hierarchy
        hierarchy child
    parent_indices: 1d ndarray
        hierarchy parent (`child.hparent`) indices to map to child

    Returns
    -------
    child_indices: 1d ndarray
        equivalent of `parent_indices` in `child` dataset
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
    """Map root RTDCBase event indices to child RTDCBase

    Parameters
    ----------
    child: RTDCBase
        hierarchy child to map to
    root_indices: 1d ndarray
        hierarchy root indices to map

    Returns
    -------
    child_indices: 1d ndarray
        child indices
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
