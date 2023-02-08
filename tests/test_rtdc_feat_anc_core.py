import numpy as np
import pytest

import dclab
from dclab.rtdc_dataset.feat_anc_core.ancillary_feature import (
    AncillaryFeature, BadFeatureSizeWarning)

from helper_methods import example_data_dict, retrieve_data, calltracker


@pytest.fixture
def fake_af_cleanup_fixture():
    """Fixture used to setup and cleanup some fake ancillary features"""
    # code run before the test
    # set the method calls of our shared method to zero (just in case)
    shared_fake_af_method.calls = 0
    # setup fake ancillary features
    af1, af2 = setup_fake_af()
    # variables yielded (input) to the test, then the test is run
    yield af1, af2
    # code run after the test
    # set the method calls of our shared method to zero
    shared_fake_af_method.calls = 0
    # remove our fake ancillary feature instances from the registry
    AncillaryFeature.features.remove(af1)
    AncillaryFeature.features.remove(af2)
    AncillaryFeature.feature_names.remove(af1.feature_name)
    AncillaryFeature.feature_names.remove(af2.feature_name)


@calltracker
def shared_fake_af_method(rtdc_ds):
    """An example of an `AncillaryFeature.method` that is used to calculate two
    different ancillary features.
    """
    cw = rtdc_ds.config["setup"]["channel width"]
    data_dict = {
        "userdef1": np.arange(1, len(rtdc_ds) + 1) * cw,
        "userdef2": np.arange(1, len(rtdc_ds) + 1) * cw * 2
    }
    return data_dict


def setup_fake_af():
    """Creates some example ancillary features"""
    af1 = AncillaryFeature(feature_name="userdef1",
                           method=shared_fake_af_method,
                           req_config=[["setup", ["channel width"]]])
    af2 = AncillaryFeature(feature_name="userdef2",
                           method=shared_fake_af_method,
                           req_config=[["setup", ["channel width"]]])
    return af1, af2


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_af_0basic():
    pytest.importorskip("nptdms")
    ds = dclab.new_dataset(retrieve_data("fmt-tdms_fl-image_2016.zip"))
    for cc in ['fl1_pos',
               'frame',
               'size_x',
               'size_y',
               'contour',
               'area_cvx',
               'circ',
               'image',
               'trace',
               'fl1_width',
               'nevents',
               'pos_x',
               'pos_y',
               'fl1_area',
               'fl1_max',
               ]:
        assert cc in ds

    # ancillaries
    for cc in ["deform",
               "area_um",
               "aspect",
               "frame",
               "index",
               "time",
               ]:
        assert cc in ds


def test_af_0error():
    keys = ["circ"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)
    try:
        ds["unknown_column"]
    except KeyError:
        pass
    else:
        raise ValueError("Should have raised KeyError!")


def test_af_aspect():
    pytest.importorskip("nptdms")
    # Aspect ratio of the data
    ds = dclab.new_dataset(retrieve_data("fmt-tdms_fl-image-bright_2017.zip"))
    aspect = ds["aspect"]
    assert np.sum(aspect > 1) == 904
    assert np.sum(aspect < 1) == 48


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_af_area_ratio():
    pytest.importorskip("nptdms")
    ds = dclab.new_dataset(retrieve_data("fmt-tdms_fl-image_2016.zip"))
    comp_ratio = ds["area_ratio"]
    # The convex area is always >= the raw area
    assert np.all(comp_ratio >= 1)
    assert np.allclose(comp_ratio[0], 1.0196464)


def test_af_deform():
    keys = ["circ"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)
    assert np.allclose(ds["deform"], 1 - ds["circ"])


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.parametrize("path", [
    "fmt-tdms_minimal_2016.zip",
    "fmt-tdms_fl-image_2016.zip",
    "fmt-tdms_fl-image-bright_2017.zip",
    "fmt-tdms_fl-image-large-fov_2017.zip",
    "fmt-tdms_shapein-2.0.1-no-image_2017.zip",
])
def test_af_method_called_once_with_shared_pipeline(
        fake_af_cleanup_fixture, path):
    """Verifies that when an `AncillaryFeature.method` is shared between
    ancillary features, the `method` itself is only called once. All ancillary
    features that share the `method` will be populated in this single call.
    """
    pytest.importorskip("nptdms")
    af1, af2 = fake_af_cleanup_fixture
    ds = dclab.new_dataset(retrieve_data(path))
    assert af1.feature_name not in ds.features_innate
    assert af2.feature_name not in ds.features_innate
    assert af1.feature_name in ds
    assert af2.feature_name in ds

    # handle tdms "time" `AncillaryFeature`
    tdms_time_feature = False
    feats = [af1.feature_name, af2.feature_name]
    if ds.format == "tdms":
        tdms_time_feature = True
        feats.append("time")

    assert shared_fake_af_method.calls == 0
    assert len(ds._ancillaries) == 0 + tdms_time_feature

    _ = ds[af1.feature_name]
    assert shared_fake_af_method.calls == 1
    assert len(ds._ancillaries) == 2 + tdms_time_feature
    assert all(k in ds._ancillaries for k in feats)

    _ = ds[af2.feature_name]
    assert shared_fake_af_method.calls == 1, "method is not called again"
    assert len(ds._ancillaries) == 2 + tdms_time_feature


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.parametrize("path", [
    "fmt-tdms_minimal_2016.zip",
    "fmt-tdms_fl-image_2016.zip",
    "fmt-tdms_fl-image-bright_2017.zip",
    "fmt-tdms_fl-image-large-fov_2017.zip",
    "fmt-tdms_shapein-2.0.1-no-image_2017.zip",
])
def test_af_recomputed_on_hash_change(fake_af_cleanup_fixture, path):
    """Check whether features are recomputed when the hash changes"""
    pytest.importorskip("nptdms")
    af1, af2 = fake_af_cleanup_fixture
    ds = dclab.new_dataset(retrieve_data(path))
    assert af1.feature_name not in ds.features_innate
    assert af2.feature_name not in ds.features_innate
    assert af1.feature_name in ds
    assert af2.feature_name in ds

    ud1a = ds[af1.feature_name]
    ud2a = ds[af2.feature_name]

    ds.config["setup"]["channel width"] *= 1.1

    ud1b = ds[af1.feature_name]
    ud2b = ds[af2.feature_name]

    assert np.all(ud1a != ud1b)
    assert np.allclose(ud1a * 1.1, ud1b)
    assert np.all(ud2a != ud2b)
    assert np.allclose(ud2a * 1.1, ud2b)


def test_af_time():
    pytest.importorskip("nptdms")
    ds = dclab.new_dataset(retrieve_data("fmt-tdms_minimal_2016.zip"))
    time = ds["time"]
    frame = ds["frame"]
    fr = ds.config["imaging"]["frame rate"]
    assert np.allclose(time[0], frame[0] / fr, atol=0, rtol=1e-7)
    assert np.allclose(time[1], 82.526, atol=0, rtol=1e-7)
    assert np.all(np.diff(time) > 0)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_af_warning_from_check_data_size():
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    with dclab.new_dataset(h5path) as ds:
        data = np.arange(len(ds)//2)
        data_dict = {"name1": data}
        with pytest.warns(BadFeatureSizeWarning):
            AncillaryFeature.check_data_size(ds, data_dict)
