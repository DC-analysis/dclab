
import numpy as np

import dclab

from helper_methods import example_data_dict


def test_downsample_index():
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)

    ds.apply_filter()
    x, y, index = ds.get_downsampled_scatter(xax="area_um",
                                             yax="deform",
                                             downsample=100,
                                             ret_mask=True)
    assert np.all(x == ds["area_um"][index])
    assert np.all(y == ds["deform"][index])

    # also with log scale
    x2, y2, index2 = ds.get_downsampled_scatter(xax="area_um",
                                                yax="deform",
                                                downsample=100,
                                                xscale="log",
                                                ret_mask=True)
    assert np.all(x2 == ds["area_um"][index2])
    assert np.all(y2 == ds["deform"][index2])


def test_downsample_log():
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)

    ds.apply_filter()
    xlin, ylin = ds.get_downsampled_scatter(downsample=100,
                                            xscale="linear",
                                            yscale="linear")
    xlog, ylog = ds.get_downsampled_scatter(downsample=100,
                                            xscale="log",
                                            yscale="log")
    assert not np.all(xlin == xlog)
    assert not np.all(ylin == ylog)
    for x in xlog:
        assert x in ds["area_um"]
    for y in ylog:
        assert y in ds["deform"]


def test_downsample_log_invalid():
    ddict = {"area_um": np.linspace(-2, 10, 100),
             "deform": np.linspace(.1, .5, 100)}
    ds = dclab.new_dataset(ddict)
    ds.apply_filter()
    xlog, ylog = ds.get_downsampled_scatter(downsample=99,
                                            xscale="log",
                                            yscale="log",
                                            remove_invalid=False,
                                            )
    for x in xlog:
        assert x in ds["area_um"]
    for y in ylog:
        assert y in ds["deform"]

    assert xlog.size == 99
    assert ylog.size == 99
    assert xlog.min() < 0


def test_downsample_log_invalid_index():
    ddict = {"area_um": np.linspace(-2, 10, 100),
             "deform": np.linspace(.1, .5, 100)}
    ds = dclab.new_dataset(ddict)
    ds.apply_filter()
    x, y, index = ds.get_downsampled_scatter(xax="area_um",
                                             yax="deform",
                                             downsample=99,
                                             xscale="log",
                                             yscale="log",
                                             remove_invalid=False,
                                             ret_mask=True
                                             )
    assert np.all(x == ds["area_um"][index])
    assert np.all(y == ds["deform"][index])


def test_downsample_log_invalid_removed():
    ddict = {"area_um": np.array([-100, 0, 100, 200, 300, 400, 500]),
             "deform": np.array([.1, .2, .3, .4, .5, 6, np.nan])}
    ds = dclab.new_dataset(ddict)
    ds.apply_filter()
    xlog, ylog = ds.get_downsampled_scatter(downsample=6,
                                            xscale="log",
                                            yscale="log",
                                            remove_invalid=True,
                                            )
    assert xlog.size == 4
    assert ylog.size == 4
    assert np.all(xlog == np.array([100, 200, 300, 400]))
    assert np.all(ylog == np.array([.3, .4, .5, 6]))


def test_downsample_log_invalid_removed_index():
    ddict = {"area_um": np.array([-100, 0, 100, 200, 300, 400, 500]),
             "deform": np.array([.1, .2, .3, .4, .5, 6, np.nan])}
    ds = dclab.new_dataset(ddict)
    ds.apply_filter()
    x, y, index = ds.get_downsampled_scatter(xax="area_um",
                                             yax="deform",
                                             downsample=6,
                                             xscale="log",
                                             yscale="log",
                                             remove_invalid=True,
                                             ret_mask=True
                                             )
    assert np.all(x == ds["area_um"][index])
    assert np.all(y == ds["deform"][index])


def test_downsample_none():
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)

    ds.apply_filter()
    _, _, idx = ds.get_downsampled_scatter(downsample=0, ret_mask=True)
    assert np.sum(idx) == 8472


def test_downsample_none2():
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)

    filtflt = {"enable filters": False}

    cfg = {"filtering": filtflt}
    ds.config.update(cfg)
    ds.apply_filter()
    _, _, idx = ds.get_downsampled_scatter(downsample=100, ret_mask=True)

    assert np.sum(idx) == 100
    assert np.sum(ds.filter.all) == 8472

    filtflt["enable filters"] = True
    ds.config.update(cfg)
    ds.apply_filter()
    _, _, idx = ds.get_downsampled_scatter(downsample=100, ret_mask=True)

    assert np.sum(idx) == 100
    assert np.sum(ds.filter.all) == 8472


def test_downsample_up():
    """
    Likely causes removal of too many points and requires
    re-inserting them.
    """
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=10000, keys=keys)
    ds = dclab.new_dataset(ddict)

    ds.apply_filter()
    _, _, idx = ds.get_downsampled_scatter(downsample=9999, ret_mask=True)
    assert np.sum(idx) == 9999
    ds.get_downsampled_scatter(downsample=9999)


def test_downsample_yes():
    """Simple downsampling test"""
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=8472, keys=keys)
    ds = dclab.new_dataset(ddict)

    ds.apply_filter()
    _, _, idx = ds.get_downsampled_scatter(downsample=100, ret_mask=True)
    assert np.sum(idx) == 100
    ds.get_downsampled_scatter(downsample=100)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
