import numpy as np

import dclab
from dclab.rtdc_dataset.ancillaries import AncillaryFeature

from helper_methods import example_data_dict, retrieve_data, \
    example_data_sets


def calltracker(func):
    """ Decorator to track how many times a function is called """
    def wrapped(*args, **kwargs):
        wrapped.calls += 1
        return func(*args, **kwargs)
    wrapped.calls = 0
    return wrapped


@calltracker
def example_shared_af_method(rtdc_ds):
    cw = rtdc_ds.config["setup"]["channel width"]
    data_dict = {
        "userdef1": np.arange(1, len(rtdc_ds) + 1) * cw,
        "userdef2": np.arange(1, len(rtdc_ds) + 1) * cw * 2
    }
    return data_dict


def setup_fake_af(feature_name1, feature_name2,
                  path="rtdc_data_hdf5_rtfdc.zip"):
    af1 = AncillaryFeature(feature_name=feature_name1,
                           method=example_shared_af_method,
                           req_config=[["setup", ["channel width"]]])
    af2 = AncillaryFeature(feature_name=feature_name2,
                           method=example_shared_af_method,
                           req_config=[["setup", ["channel width"]]])

    ds = dclab.new_dataset(retrieve_data(path))
    assert feature_name1 not in ds.features_innate
    assert feature_name2 not in ds.features_innate
    assert feature_name1 in ds
    assert feature_name2 in ds
    return ds, af1, af2


def cleanup_fake_af(af1, af2, path="rtdc_data_hdf5_rtfdc.zip"):
    af1_feature_name, af2_feature_name = af1.feature_name, af2.feature_name
    AncillaryFeature.features.remove(af1)
    AncillaryFeature.features.remove(af2)
    AncillaryFeature.feature_names.remove(af1_feature_name)
    AncillaryFeature.feature_names.remove(af2_feature_name)
    # make sure cleanup worked
    ds2 = dclab.new_dataset(retrieve_data(path))
    assert af1_feature_name not in ds2
    assert af2_feature_name not in ds2
    # cleanup the calltracker
    example_shared_af_method.calls = 0



def test_af_0basic():
    ds = dclab.new_dataset(retrieve_data(example_data_sets[1]))
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
    # Aspect ratio of the data
    ds = dclab.new_dataset(retrieve_data("rtdc_data_traces_video_bright.zip"))
    aspect = ds["aspect"]
    assert np.sum(aspect > 1) == 904
    assert np.sum(aspect < 1) == 48


def test_af_area_ratio():
    ds = dclab.new_dataset(retrieve_data("rtdc_data_traces_video.zip"))
    comp_ratio = ds["area_ratio"]
    # The convex area is always >= the raw area
    assert np.all(comp_ratio >= 1)
    assert np.allclose(comp_ratio[0], 1.0196464)


def test_af_deform():
    keys = ["circ"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)
    assert np.allclose(ds["deform"], 1 - ds["circ"])


def test_af_shared_af_method_pipeline_called_once_hdf5():
    """Uses `setup_cleanup_fake_ancillary_features` as a fixture"""
    path = "rtdc_data_hdf5_rtfdc.zip"
    feature_name1, feature_name2 = "userdef1", "userdef2"
    ds, af1, af2 = setup_fake_af(feature_name1, feature_name2, path)

    assert example_shared_af_method.calls == 0
    assert len(ds._ancillaries) == 0

    _ = ds[feature_name1]
    assert example_shared_af_method.calls == 1
    assert len(ds._ancillaries) == 2

    _ = ds[feature_name2]
    assert example_shared_af_method.calls == 1, "method is not called again"
    assert len(ds._ancillaries) == 2
    feats = [feature_name1, feature_name2]
    for af_key in ds._ancillaries:
        assert af_key in feats
    cleanup_fake_af(af1, af2, path)


def test_af_shared_af_method_pipeline_called_once_tdms():
    """Calling ancillary features will automatically populate other features
    that share the same method.
    """
    path = "rtdc_data_traces_video_bright.zip"
    feature_name1, feature_name2 = "userdef1", "userdef2"
    ds, af1, af2 = setup_fake_af(feature_name1, feature_name2, path)

    assert example_shared_af_method.calls == 0
    assert len(ds._ancillaries) == 1  # time is populated
    # both "bright_avg" and "bright_sd" will be populated by one call
    _ = ds[feature_name1]
    assert example_shared_af_method.calls == 1
    assert len(ds._ancillaries) == 3

    _ = ds[feature_name2]
    assert example_shared_af_method.calls == 1, "method is not called again"
    assert len(ds._ancillaries) == 3

    feats = ["time", feature_name1, feature_name2]
    for af_key in ds._ancillaries:
        assert af_key in feats
    cleanup_fake_af(af1, af2, path)


def test_af_recomputed_on_hash_change():
    """Check whether features are recomputed when the hash changes.
    Uses `setup_cleanup_fake_ancillary_features` as a fixture.
    """
    path = "rtdc_data_hdf5_rtfdc.zip"
    feature_name1, feature_name2 = "userdef1", "userdef2"
    ds, af1, af2 = setup_fake_af(feature_name1, feature_name2, path)

    ud1a = ds[feature_name1]
    ud2a = ds[feature_name2]

    ds.config["setup"]["channel width"] *= 1.1

    ud1b = ds[feature_name1]
    ud2b = ds[feature_name2]

    assert np.all(ud1a != ud1b)
    assert np.allclose(ud1a * 1.1, ud1b)
    assert np.all(ud2a != ud2b)
    assert np.allclose(ud2a * 1.1, ud2b)
    cleanup_fake_af(af1, af2, path)


def test_af_time():
    ds = dclab.new_dataset(retrieve_data("rtdc_data_minimal.zip"))
    tt = ds["time"]
    assert tt[0] == 0
    assert np.allclose(tt[1], 0.0385)
    assert np.all(np.diff(tt) > 0)


# def test_af_mock_with_patch(monkeypatch):

    # def calltracker(func):
    #     """ Decorator to track how many times a function is called """
    #     def wrap_mock():
    #         mock_run = Mock()
    #         return mock_run
    #     return wrap_mock

    # old_get_bright = dclab.features.bright.get_bright

    # @calltracker
    # def wrapper_get_bright(mm):
    #     old_get_bright(mask=mm["mask"], image=mm["image"], ret_data="avg,sd")
    # mock_run = Mock()
    # monkeypatch.setattr(
    #     'dclab.rtdc_dataset.core.RTDCBase.__getitem__.compute', mock_run)
    #
    # ds = dclab.new_dataset(retrieve_data(
    #     "rtdc_data_traces_video_bright.zip"))
    # _ = ds["bright_sd"]
    # # _ = ds["bright_avg"]
    # mock_run.assert_called_once()
    # mock_run.reset_mock()


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
