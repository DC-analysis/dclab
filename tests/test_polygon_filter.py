import os
import tempfile

import numpy as np
import pytest

import dclab

from helper_methods import example_data_dict


filter_data = """[Polygon 00000000]
X Axis = aspect
Y Axis = tilt
Name = polygon filter 0
point00000000 = 6.344607717656481e-03 7.703315881326352e-01
point00000001 = 7.771010748302133e-01 7.452006980802792e-01
point00000002 = 8.025596093384512e-01 6.806282722513089e-03
point00000003 = 6.150993521573982e-01 1.015706806282723e-03
"""


@pytest.fixture(autouse=True)
def run_around_tests():
    dclab.PolygonFilter.clear_all_filters()
    yield
    dclab.PolygonFilter.clear_all_filters()


@pytest.mark.filterwarnings('ignore::dclab.polygon_filter.'
                            + 'FilterIdExistsWarning')
def test_import():
    ddict = example_data_dict(size=1000, keys=["aspect", "tilt"])
    ds = dclab.new_dataset(ddict)

    # save polygon data
    _fd, tf = tempfile.mkstemp(prefix="dclab_polgyon_test")
    with open(tf, "w") as fd:
        fd.write(filter_data)

    # Add polygon filter
    pf = dclab.PolygonFilter(filename=tf)
    ds.polygon_filter_add(pf)

    ds.apply_filter()

    assert np.sum(ds.filter.all) == 330

    dclab.PolygonFilter.import_all(tf)

    assert len(dclab.PolygonFilter.instances) == 2

    # Import multiples
    b = filter_data
    b = b.replace("Polygon 00000000", "Polygon 00000001")
    with open(tf, "a") as fd:
        fd.write(b)
    dclab.PolygonFilter.import_all(tf)

    # Import previously saved
    dclab.PolygonFilter.save_all(tf)
    dclab.PolygonFilter.import_all(tf)

    assert len(dclab.PolygonFilter.instances) == 10


def test_import_and_apply_to_hierarchy_child():
    ddict = example_data_dict(size=1000, keys=["aspect", "tilt"])
    ds = dclab.new_dataset(ddict)
    ch1 = dclab.new_dataset(ds)

    # save polygon data
    _fd, tf = tempfile.mkstemp(prefix="dclab_polgyon_test")
    with open(tf, "w") as fd:
        fd.write(filter_data)

    # Add polygon filter
    pf = dclab.PolygonFilter(filename=tf)
    ch1.polygon_filter_add(pf)

    ch1.apply_filter()

    ch2 = dclab.new_dataset(ch1)

    assert np.sum(ch1.filter.all) == 330

    assert len(ch2) == 330
    assert len(ch2["aspect"]) == 330


def test_import_custom_unique_id():
    # save polygon data
    _fd, tf = tempfile.mkstemp(prefix="dclab_polgyon_test")
    with open(tf, "w") as fd:
        fd.write(filter_data)

    pf = dclab.PolygonFilter(filename=tf)
    pf2 = dclab.PolygonFilter(filename=tf, unique_id=10)

    assert pf.unique_id == 0
    assert pf2.unique_id == 10


def test_invert():
    ddict = example_data_dict(size=1234, keys=["aspect", "tilt"])
    ds = dclab.new_dataset(ddict)
    # points of polygon filter
    points = [[np.min(ddict["aspect"]), np.min(ddict["tilt"])],
              [np.min(ddict["aspect"]), np.max(ddict["tilt"])],
              [np.average(ddict["aspect"]), np.max(ddict["tilt"])],
              [np.average(ddict["aspect"]), np.min(ddict["tilt"])],
              ]
    filt1 = dclab.PolygonFilter(axes=["aspect", "tilt"],
                                points=points,
                                inverted=False)
    ds.polygon_filter_add(filt1)
    assert [0] == ds.config["filtering"]["polygon filters"]
    n1 = np.sum(ds.filter.all)
    ds.apply_filter()
    n2 = np.sum(ds.filter.all)
    assert n1 != n2
    filt2 = dclab.PolygonFilter(axes=["aspect", "tilt"],
                                points=points,
                                inverted=True)
    ds.polygon_filter_add(filt2)
    assert [0, 1] == ds.config["filtering"]["polygon filters"]
    ds.apply_filter()
    assert np.sum(ds.filter.all) == 0, "inverted+normal filter filters all"


def test_invert_copy():
    ddict = example_data_dict(size=1234, keys=["aspect", "tilt"])
    ds = dclab.new_dataset(ddict)
    # points of polygon filter
    points = [[np.min(ddict["aspect"]), np.min(ddict["tilt"])],
              [np.min(ddict["aspect"]), np.max(ddict["tilt"])],
              [np.average(ddict["aspect"]), np.max(ddict["tilt"])],
              [np.average(ddict["aspect"]), np.min(ddict["tilt"])],
              ]
    filt1 = dclab.PolygonFilter(axes=["aspect", "tilt"],
                                points=points,
                                inverted=False)
    ds.polygon_filter_add(filt1)
    assert [0] == ds.config["filtering"]["polygon filters"]
    n1 = np.sum(ds.filter.all)
    ds.apply_filter()
    n2 = np.sum(ds.filter.all)
    assert n1 != n2
    filt2 = filt1.copy(invert=True)
    ds.polygon_filter_add(filt2)
    assert [0, 1] == ds.config["filtering"]["polygon filters"]
    ds.apply_filter()
    assert np.sum(ds.filter.all) == 0, "inverted+normal filter filters all"


@pytest.mark.filterwarnings('ignore::dclab.polygon_filter.'
                            + 'FilterIdExistsWarning')
def test_invert_saveload():
    ddict = example_data_dict(size=1234, keys=["aspect", "tilt"])
    # points of polygon filter
    points = [[np.min(ddict["aspect"]), np.min(ddict["tilt"])],
              [np.min(ddict["aspect"]), np.max(ddict["tilt"])],
              [np.average(ddict["aspect"]), np.max(ddict["tilt"])],
              [np.average(ddict["aspect"]), np.min(ddict["tilt"])],
              ]
    filt1 = dclab.PolygonFilter(axes=["aspect", "tilt"],
                                points=points,
                                inverted=True)
    name = tempfile.mktemp(prefix="test_dclab_polygon_")
    filt1.save(name)
    filt2 = dclab.PolygonFilter(filename=name)
    assert filt2 == filt1

    filt3 = dclab.PolygonFilter(axes=["aspect", "tilt"],
                                points=points,
                                inverted=False)
    try:
        os.remove(name)
    except OSError:
        pass

    name = tempfile.mktemp(prefix="test_dclab_polygon_")
    filt3.save(name)
    filt4 = dclab.PolygonFilter(filename=name)
    assert filt4 == filt3


def test_inverted_wrong():
    ddict = example_data_dict(size=1234, keys=["aspect", "tilt"])
    # points of polygon filter
    points = [[np.min(ddict["aspect"]), np.min(ddict["tilt"])],
              [np.min(ddict["aspect"]), np.max(ddict["tilt"])],
              [np.average(ddict["aspect"]), np.max(ddict["tilt"])],
              [np.average(ddict["aspect"]), np.min(ddict["tilt"])],
              ]
    with pytest.raises(dclab.polygon_filter.PolygonFilterError,
                       match="must be boolean"):
        dclab.PolygonFilter(axes=["aspect", "tilt"],
                            points=points,
                            inverted=0)


def test_nofile_copy():
    a = dclab.PolygonFilter(axes=("tilt", "aspect"),
                            points=[[0, 1], [1, 1]])
    a.copy()


def test_remove():
    _fd, tf = tempfile.mkstemp(prefix="dclab_polgyon_test")
    with open(tf, "w") as fd:
        fd.write(filter_data)

    # Add polygon filter
    pf = dclab.PolygonFilter(filename=tf)

    dclab.PolygonFilter.remove(pf.unique_id)
    assert len(dclab.PolygonFilter.instances) == 0


@pytest.mark.filterwarnings('ignore::dclab.polygon_filter.'
                            + 'FilterIdExistsWarning')
def test_save():
    _fd, tf = tempfile.mkstemp(prefix="dclab_polgyon_test")
    with open(tf, "w") as fd:
        fd.write(filter_data)

    # Add polygon filter
    pf = dclab.PolygonFilter(filename=tf)

    _fd, tf2 = tempfile.mkstemp(prefix="dclab_polgyon_test")
    with open(tf2, "w") as fd:
        fd.write(filter_data)
        pf.save(tf2, ret_fobj=True)
        pf2 = dclab.PolygonFilter(filename=tf2)
        assert np.allclose(pf.points, pf2.points)

    _fd, tf3 = tempfile.mkstemp(prefix="dclab_polgyon_test")
    dclab.PolygonFilter.save_all(tf3)
    pf.save(tf3, ret_fobj=False)

    # ensure backwards compatibility: the names of the
    # three filters should be the same
    names = dclab.polygon_filter.get_polygon_filter_names()
    assert len(names) == 2
    assert names.count(names[0]) == 2


@pytest.mark.filterwarnings('ignore::dclab.polygon_filter.'
                            + 'FilterIdExistsWarning')
def test_save_multiple():
    _fd, tf = tempfile.mkstemp(prefix="dclab_polgyon_test")
    with open(tf, "w") as fd:
        fd.write(filter_data)

    # Add polygon filter
    pf = dclab.PolygonFilter(filename=tf)

    _fd, tf2 = tempfile.mkstemp(prefix="dclab_polgyon_test")
    with open(tf2, "a") as fd:
        pf.save(fd)
        pf2 = dclab.PolygonFilter(filename=tf2)
        assert np.allclose(pf.points, pf2.points)


def test_state():
    _fd, tf = tempfile.mkstemp(prefix="dclab_polgyon_test")
    with open(tf, "w") as fd:
        fd.write(filter_data)

    # Add polygon filter
    pf = dclab.PolygonFilter(filename=tf)

    state = pf.__getstate__()
    assert state["name"] == "polygon filter 0"
    assert state["axis x"] == "aspect"
    assert state["axis y"] == "tilt"
    assert np.allclose(state["points"][0][0], 6.344607717656481e-03)
    assert np.allclose(state["points"][3][1], 1.015706806282723e-03)
    assert not state["inverted"]

    state["name"] = "peter"
    state["axis x"] = "tilt"
    state["axis y"] = "aspect"
    state["points"][0][0] = 1
    state["inverted"] = True

    pf.__setstate__(state)

    assert pf.name == "peter"
    assert pf.axes[0] == "tilt"
    assert pf.axes[1] == "aspect"
    assert np.allclose(pf.points[0, 0], 1)
    assert pf.inverted


@pytest.mark.filterwarnings('ignore::dclab.polygon_filter.'
                            + 'FilterIdExistsWarning')
def test_unique_id():
    _fd, tf = tempfile.mkstemp(prefix="dclab_polgyon_test")
    with open(tf, "w") as fd:
        fd.write(filter_data)

    # Add polygon filter
    pf = dclab.PolygonFilter(filename=tf, unique_id=2)
    pf2 = dclab.PolygonFilter(filename=tf, unique_id=2)
    assert pf.unique_id != pf2.unique_id


@pytest.mark.filterwarnings('ignore::dclab.polygon_filter.'
                            + 'FilterIdExistsWarning')
def test_with_rtdc_data_set():
    ddict = example_data_dict(size=821, keys=["aspect", "tilt"])
    ds = dclab.new_dataset(ddict)

    # save polygon data
    _fd, tf = tempfile.mkstemp(prefix="dclab_polgyon_test")
    with open(tf, "w") as fd:
        fd.write(filter_data)
    pf = dclab.PolygonFilter(filename=tf)
    pf2 = dclab.PolygonFilter(filename=tf)

    ds.polygon_filter_add(pf)
    ds.polygon_filter_add(1)

    ds.polygon_filter_rm(0)
    ds.polygon_filter_rm(pf2)


def test_wrong_load_key():
    # save polygon data
    _fd, tf = tempfile.mkstemp(prefix="dclab_polgyon_test")
    with open(tf, "w") as fd:
        fd.write(filter_data + "peter=4\n")

    with pytest.raises(KeyError, match="Unknown variable: peter = 4"):
        dclab.PolygonFilter(filename=tf)
