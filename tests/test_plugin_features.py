import pathlib
import h5py
import numpy as np
import pytest
from scipy.ndimage.filters import gaussian_filter

import dclab
from dclab.rtdc_dataset.plugins.plugin_feature import (
    PlugInFeature, import_plugin_feature_script,
    remove_plugin_feature, remove_all_plugin_features,
    PluginImportError)
from dclab.rtdc_dataset.ancillaries.ancillary_feature import (
    BadFeatureSizeWarning)

from helper_methods import retrieve_data

data_dir = pathlib.Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def cleanup_plugin_features():
    """Fixture used to cleanup plugin feature tests"""
    # code run before the test
    pass
    # then the test is run
    yield
    # code run after the test
    # remove our test plugin examples
    remove_all_plugin_features()


def compute_single_plugin_feature(rtdc_ds):
    """Basic plugin method"""
    circ_per_area = rtdc_ds["circ"] / rtdc_ds["area_um"]
    return circ_per_area


def compute_multiple_plugin_features(rtdc_ds):
    """Basic plugin method with dictionary returned"""
    circ_per_area = rtdc_ds["circ"] / rtdc_ds["area_um"]
    circ_times_area = rtdc_ds["circ"] * rtdc_ds["area_um"]
    return {"circ_per_area": circ_per_area, "circ_times_area": circ_times_area}


def compute_non_scalar_plugin_feature(rtdc_ds):
    """Basic non-scalar plugin method"""
    image_gauss_filter = gaussian_filter(rtdc_ds["image"], sigma=(0, 1, 1))
    return {"image_gauss_filter": image_gauss_filter}


def example_plugin_info_single_feature():
    """plugin info for a single feature"""
    info = {
        "method": compute_single_plugin_feature,
        "description": "This plugin will compute a feature",
        "long description": "Even longer description that "
                            "can span multiple lines",
        "feature names": ["circ_per_area"],
        "feature labels": ["Circularity per Area"],
        "features required": ["circ", "area_um"],
        "config required": [],
        "method check required": lambda x: True,
        "scalar feature": [True],
        "version": "0.1.0",
    }
    return info


def example_plugin_info_multiple_feature():
    """plugin info for multiple features"""
    info = {
        "method": compute_multiple_plugin_features,
        "description": "This plugin will compute some features",
        "long description": "Even longer description that "
                            "can span multiple lines",
        "feature names": ["circ_per_area", "circ_times_area"],
        "feature labels": ["Circularity per Area", "Circularity times Area"],
        "features required": ["circ", "area_um"],
        "config required": [],
        "method check required": lambda x: True,
        "scalar feature": [True, True],
        "version": "0.1.0",
    }
    return info


def example_plugin_info_non_scalar_feature():
    """plugin info for non-scalar feature"""
    info = {
        "method": compute_non_scalar_plugin_feature,
        "description": "This plugin will compute a non-scalar feature",
        "long description": "This non-scalar feature is a Gaussian filter of "
                            "the image",
        "feature names": ["image_gauss_filter"],
        "feature labels": ["Gaussian Filtered Image"],
        "features required": ["image"],
        "config required": [],
        "method check required": lambda x: True,
        "scalar feature": [False],
        "version": "0.1.0",
    }
    return info


def compute_with_user_section(rtdc_ds):
    """setup a plugin method that uses user config section

    The "user:n_constrictions" metadata must be set
    """
    nc = rtdc_ds.config["user"]["n_constrictions"]
    assert isinstance(nc, int), (
        '"n_constrictions" should be an integer value.')
    area_of_region = rtdc_ds["area_um"] * nc
    return {"area_of_region": area_of_region}


def test_pf_attribute_ancill_info():
    """Check the plugin feature attribute input to AncillaryFeature"""
    info = example_plugin_info_single_feature()
    pf = PlugInFeature("circ_per_area", info)
    assert pf.plugin_feature_info["feature name"] == "circ_per_area"
    assert pf.plugin_feature_info["method"] is compute_single_plugin_feature
    assert pf.plugin_feature_info["config required"] == []
    assert pf.plugin_feature_info["features required"] == ["circ", "area_um"]


def test_pf_attribute_plugin_feature_info():
    """Check the plugin feature info attribute"""
    info = example_plugin_info_single_feature()
    # comparing lambda functions fails due to differing memory locations
    info.pop("method check required")
    pf = PlugInFeature("circ_per_area", info)
    pf.plugin_feature_info.pop("method check required")
    plugin_feature_info = {
        "method": compute_single_plugin_feature,
        "description": "This plugin will compute a feature",
        "long description": "Even longer description that "
                            "can span multiple lines",
        "feature name": "circ_per_area",
        "feature label": "Circularity per Area",
        "features required": ["circ", "area_um"],
        "config required": [],
        "scalar feature": True,
        "version": "0.1.0",
        "plugin path": None,
    }
    assert pf.plugin_feature_info == plugin_feature_info


def test_pf_attributes():
    """Check the plugin feature attributes"""
    plugin_path = data_dir / "plugin_test_example.py"
    plugin_list = dclab.load_plugin_feature(plugin_path)
    pf1, pf2 = plugin_list
    plugin_file_info = import_plugin_feature_script(plugin_path)

    assert pf1.feature_name == pf1.feature_name == \
           plugin_file_info["feature names"][0]
    assert pf2.feature_name == pf2.feature_name == \
           plugin_file_info["feature names"][1]

    assert plugin_path.samefile(pf1.plugin_path)
    assert plugin_path.samefile(pf1.plugin_feature_info["plugin path"])

    assert plugin_path.samefile(pf2.plugin_path)
    assert plugin_path.samefile(pf2.plugin_feature_info["plugin path"])

    assert pf1._original_info == plugin_file_info
    assert pf2._original_info == plugin_file_info


def test_pf_attributes_af_inherited():
    """Check the plugin feature attributes inherited from AncillaryFeature"""
    plugin_path = data_dir / "plugin_test_example.py"
    plugin_list = dclab.load_plugin_feature(plugin_path)
    pf, _ = plugin_list
    plugin_file_info = import_plugin_feature_script(plugin_path)

    assert pf.feature_name == plugin_file_info["feature names"][0]
    assert pf.method == plugin_file_info["method"]
    assert pf.req_config == plugin_file_info["config required"]
    assert pf.req_features == plugin_file_info["features required"]
    assert pf.req_func == plugin_file_info["method check required"]
    assert pf.priority == 0


def test_pf_bad_plugin_feature_name_list():
    """Basic test of a bad feature name for PlugInFeature"""
    info = example_plugin_info_single_feature()
    info["feature names"] = "Peter-Pan's Best Friend!"
    with pytest.raises(ValueError, match="must be a list, got"):
        PlugInFeature("Peter-Pan's Best Friend!", info)


def test_pf_bad_plugin_feature_name():
    """Basic test of a bad feature name for PlugInFeature"""
    info = example_plugin_info_single_feature()
    info["feature names"] = ["Peter-Pan's Best Friend!"]
    with pytest.raises(ValueError, match="only contain lower-case characters"):
        PlugInFeature("Peter-Pan's Best Friend!", info)


def test_pf_exists_in_hierarchy():
    """Test that RTDCHierarchy works with PlugInFeature"""
    info = example_plugin_info_single_feature()
    pf = PlugInFeature("circ_per_area", info)
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    with dclab.new_dataset(h5path) as ds:
        assert pf.feature_name in ds
        assert dclab.dfn.feature_exists(pf.feature_name)
        child = dclab.new_dataset(ds)
        assert pf.feature_name in child


def test_pf_export_and_load():
    """Check that exported and loaded hdf5 file will keep a plugin feature"""
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    # initialize PlugInFeature instance
    info = example_plugin_info_single_feature()
    pf = PlugInFeature("circ_per_area", info)

    with dclab.new_dataset(h5path) as ds:
        # extract the feature information from the dataset
        assert pf in PlugInFeature.features
        circ_per_area = ds[pf.feature_name]

        # export the data to a new file
        expath = h5path.with_name("exported.rtdc")
        ds.export.hdf5(expath, features=ds.features_innate + [pf.feature_name])

    # make sure that worked
    with h5py.File(expath, "r") as h5:
        assert pf.feature_name in h5["events"]
        assert np.allclose(h5["events"][pf.feature_name], circ_per_area)

    # now check again with dclab
    with dclab.new_dataset(expath) as ds2:
        assert pf in PlugInFeature.features
        assert pf.feature_name in ds2
        assert pf.feature_name in ds2.features_innate
        assert np.allclose(ds2[pf.feature_name], circ_per_area)

        # and a control check
        remove_plugin_feature(pf)
        assert pf.feature_name not in ds2


def test_pf_feature_exists():
    """Basic check that the plugin feature name exists in definitions"""
    plugin_path = data_dir / "plugin_test_example.py"
    plugin_list = dclab.load_plugin_feature(plugin_path)
    assert dclab.dfn.feature_exists(plugin_list[0].feature_name)
    assert dclab.dfn.feature_exists(plugin_list[1].feature_name)


def test_pf_filtering_with_plugin_feature():
    """Filtering with plugin feature"""
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    with dclab.new_dataset(h5path) as ds:
        info = example_plugin_info_single_feature()
        pf = PlugInFeature("circ_per_area", info)

        ds.config["filtering"][f"{pf.feature_name} min"] = 0.030
        ds.config["filtering"][f"{pf.feature_name} max"] = 0.031
        ds.apply_filter()
        assert np.sum(ds.filter.all) == 1
        assert ds.filter.all[4]


def test_pf_import_plugin_info():
    """Check the plugin test example info is a dict"""
    plugin_path = data_dir / "plugin_test_example.py"
    info = import_plugin_feature_script(plugin_path)
    assert isinstance(info, dict)


def test_pf_import_plugin_info_bad_path():
    """Raise error when a bad pathname is given"""
    bad_plugin_path = "not/a/real/path/plugin.py"
    with pytest.raises(PluginImportError, match="could be not be found"):
        import_plugin_feature_script(bad_plugin_path)


def test_pf_incorrect_input_info():
    """Raise error when info is not a dictionary"""
    info = ["this", "is", "not", "a", "dict"]
    with pytest.raises(ValueError, match="must be a dict"):
        PlugInFeature("feature_1", info)


def test_pf_incorrect_input_feature_name():
    """Raise error when the feature_name doesn't match info feature name"""
    info = example_plugin_info_single_feature()
    # `feature_name` is "circ_per_area" in info
    with pytest.raises(ValueError, match="is not defined"):
        PlugInFeature("not_the_correct_name", info)


def test_pf_incorrect_input_method():
    """Raise error when method is not callable"""
    info = example_plugin_info_single_feature()
    # set `info["method"]` to something that isn't callable
    info["method"] = "this_is_a_string"
    with pytest.raises(ValueError, match="is not callable"):
        PlugInFeature("circ_per_area", info)


def test_pf_initialize_plugin_after_loading():
    """plugin feature loads correctly after feature added to hdf5 file"""
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    with dclab.new_dataset(h5path) as ds:
        circ_per_area = compute_single_plugin_feature(ds)
    with h5py.File(h5path, "a") as h5:
        h5["events"]["circ_per_area"] = circ_per_area
    with dclab.new_dataset(h5path) as ds:
        assert "circ_per_area" not in ds
        info = example_plugin_info_single_feature()
        PlugInFeature("circ_per_area", info)
        assert "circ_per_area" in ds
        assert "circ_per_area" in ds.features_innate


def test_pf_initialize_plugin_feature_single():
    """Check that single plugin feature exists independant of loaded dataset"""
    ds = dclab.new_dataset(retrieve_data("fmt-hdf5_fl_2018.zip"))
    info = example_plugin_info_single_feature()
    PlugInFeature("circ_per_area", info)
    assert "circ_per_area" in ds

    circ_per_area = ds["circ_per_area"]
    assert np.allclose(circ_per_area, ds["circ"] / ds["area_um"])

    # check that PlugInFeature exists independent of loaded ds
    ds2 = dclab.new_dataset(retrieve_data("fmt-hdf5_fl_2018.zip"))
    assert "circ_per_area" in ds2


def test_pf_initialize_plugin_feature_non_scalar():
    """Check that the non-scalar plugin feature works"""
    ds = dclab.new_dataset(retrieve_data("fmt-hdf5_fl_2018.zip"))
    info = example_plugin_info_non_scalar_feature()
    PlugInFeature("image_gauss_filter", info)
    assert "image_gauss_filter" in ds

    image_gauss_filter = ds["image_gauss_filter"]
    assert np.allclose(image_gauss_filter,
                       gaussian_filter(ds["image"], sigma=(0, 1, 1)))


def test_pf_initialize_plugin_features_multiple():
    """Check multiple plugin features exist independant of loaded dataset"""
    ds = dclab.new_dataset(retrieve_data("fmt-hdf5_fl_2018.zip"))
    assert "circ_per_area" not in ds.features_innate
    assert "circ_times_area" not in ds.features_innate
    info = example_plugin_info_multiple_feature()
    PlugInFeature("circ_per_area", info)
    PlugInFeature("circ_times_area", info)

    assert "circ_per_area" in ds
    assert "circ_times_area" in ds
    assert dclab.dfn.feature_exists("circ_per_area")
    assert dclab.dfn.feature_exists("circ_times_area")
    circ_per_area = ds["circ_per_area"]
    circ_times_area = ds["circ_times_area"]
    assert np.allclose(circ_per_area, ds["circ"] / ds["area_um"])
    assert np.allclose(circ_times_area, ds["circ"] * ds["area_um"])


def test_pf_input_no_feature_labels():
    """Check that feature labels are populated even if not given"""
    info = example_plugin_info_single_feature()
    info.pop("feature labels")
    feature_name = "circ_per_area"
    pf = PlugInFeature(feature_name, info)
    assert dclab.dfn.feature_exists(feature_name)
    label = dclab.dfn.get_feature_label(feature_name)
    assert label == "Plugin feature {}".format(feature_name)
    assert label == pf.plugin_feature_info["feature label"]


def test_pf_input_no_scalar_feature():
    """Check that scalar feature bools are populated even if not given"""
    info = example_plugin_info_single_feature()
    info.pop("scalar feature")
    pf = PlugInFeature("circ_per_area", info)
    assert pf.plugin_feature_info["scalar feature"]


def test_pf_load_plugin():
    """Basic check for loading a plugin feature via a script"""
    ds = dclab.new_dataset(retrieve_data("fmt-hdf5_fl_2018.zip"))
    assert "circ_per_area" not in ds.features_innate
    assert "circ_times_area" not in ds.features_innate
    plugin_path = data_dir / "plugin_test_example.py"
    plugin_list = dclab.load_plugin_feature(plugin_path)
    assert isinstance(plugin_list[0], PlugInFeature)
    assert isinstance(plugin_list[1], PlugInFeature)
    assert "circ_per_area" in ds
    assert "circ_times_area" in ds
    circ_per_area = ds["circ_per_area"]
    circ_times_area = ds["circ_times_area"]
    assert np.allclose(circ_per_area, ds["circ"] / ds["area_um"])
    assert np.allclose(circ_times_area, ds["circ"] * ds["area_um"])


def test_pf_minimum_info_input():
    """Only method and feature names are required to create PlugInFeature"""
    info = {"method": compute_single_plugin_feature,
            "feature names": ["circ_per_area"]}
    pf = PlugInFeature("circ_per_area", info)

    # check that all other plugin_feature_info is populated
    assert "method" in pf.plugin_feature_info
    assert callable(pf.plugin_feature_info["method"])
    assert "description" in pf.plugin_feature_info
    assert "long description" in pf.plugin_feature_info
    assert "feature name" in pf.plugin_feature_info
    assert "feature label" in pf.plugin_feature_info
    assert "features required" in pf.plugin_feature_info
    assert "config required" in pf.plugin_feature_info
    assert "method check required" in pf.plugin_feature_info
    assert "scalar feature" in pf.plugin_feature_info
    assert "version" in pf.plugin_feature_info
    assert "plugin path" in pf.plugin_feature_info


def test_pf_remove_all_plugin_features():
    """Remove all plugin features at once"""
    ds = dclab.new_dataset(retrieve_data("fmt-hdf5_fl_2018.zip"))
    assert "circ_per_area" not in ds.features_innate
    assert "circ_times_area" not in ds.features_innate
    plugin_path = data_dir / "plugin_test_example.py"
    dclab.load_plugin_feature(plugin_path)
    assert "circ_per_area" in ds
    assert "circ_times_area" in ds
    assert dclab.dfn.feature_exists("circ_per_area")
    assert dclab.dfn.feature_exists("circ_times_area")

    remove_all_plugin_features()

    assert "circ_per_area" not in ds
    assert "circ_times_area" not in ds
    assert not dclab.dfn.feature_exists("circ_per_area")
    assert not dclab.dfn.feature_exists("circ_times_area")


def test_pf_remove_plugin_feature():
    """Remove individual plugin features"""
    ds = dclab.new_dataset(retrieve_data("fmt-hdf5_fl_2018.zip"))
    assert "circ_per_area" not in ds
    assert "circ_times_area" not in ds

    plugin_path = data_dir / "plugin_test_example.py"
    plugin_list = dclab.load_plugin_feature(plugin_path)
    assert len(plugin_list) == 2
    assert "circ_per_area" in ds
    assert "circ_per_area" not in ds.features_innate
    assert "circ_times_area" in ds
    assert "circ_times_area" not in ds.features_innate
    assert dclab.dfn.feature_exists("circ_per_area")
    assert dclab.dfn.feature_exists("circ_times_area")

    remove_plugin_feature(plugin_list[0])
    remove_plugin_feature(plugin_list[1])

    assert "circ_per_area" not in ds
    assert "circ_times_area" not in ds
    assert not dclab.dfn.feature_exists("circ_per_area")
    assert not dclab.dfn.feature_exists("circ_times_area")

    with pytest.raises(TypeError,
                       match="hould be an instance of PlugInFeature"):
        not_a_plugin_instance = [4, 6, 5]
        remove_plugin_feature(not_a_plugin_instance)


def test_pf_try_existing_feature_fails():
    """An existing feature name is not allowed"""
    info = example_plugin_info_single_feature()
    info["feature names"] = ["deform"]
    with pytest.raises(ValueError, match="Feature 'deform' already exists"):
        PlugInFeature("deform", info)


def test_pf_with_empty_feature_label_string():
    """An empty string is replaced with a real feature label

    Show that an empty `feature_label` will still give a descriptive
    feature label. See `dclab.dfn._add_feature_to_definitions` for details.
    """
    info = example_plugin_info_single_feature()
    info["feature labels"] = [""]
    feature_name = "circ_per_area"
    PlugInFeature(feature_name, info)
    assert dclab.dfn.feature_exists("circ_per_area")
    label = dclab.dfn.get_feature_label("circ_per_area")
    assert label != ""
    assert label == "Plugin feature {}".format(feature_name)


def test_pf_with_feature_label():
    """Check that a plugin feature label is added to definitions"""
    info = example_plugin_info_single_feature()
    info["feature labels"] = ["Circ / Area [1/µm²]"]
    feature_name = "circ_per_area"
    PlugInFeature(feature_name, info)
    assert dclab.dfn.feature_exists("circ_per_area")
    label = dclab.dfn.get_feature_label("circ_per_area")
    assert label == "Circ / Area [1/µm²]"


def test_pf_with_no_feature_label():
    """A feature label of None is replaced with a real feature label

    Show that `feature_label=None` will still give a descriptive
    feature label. See `dclab.dfn._add_feature_to_definitions` for details.
    """
    info = example_plugin_info_single_feature()
    info["feature labels"] = [None]
    feature_name = "circ_per_area"
    PlugInFeature(feature_name, info)
    assert dclab.dfn.feature_exists("circ_per_area")
    label = dclab.dfn.get_feature_label("circ_per_area")
    assert label is not None
    assert label == "Plugin feature {}".format(feature_name)


def test_pf_with_user_config_section():
    """Use a plugin feature with the user defined config section"""
    info = {"method": compute_with_user_section,
            "feature names": ["area_of_region"],
            "config required": [["user", ["n_constrictions"]]]}
    PlugInFeature("area_of_region", info)

    ds = dclab.new_dataset(retrieve_data("fmt-hdf5_fl_2018.zip"))
    assert "area_of_region" not in ds, "not available b/c missing metadata"
    # add some metadata to the user config section
    metadata = {"channel": True,
                "n_constrictions": 3}
    ds.config["user"].update(metadata)
    assert ds.config["user"] == metadata
    assert "area_of_region" in ds, "available b/c metadata is set"

    area_of_region1 = ds["area_of_region"]
    area_of_region1_calc = (ds["area_um"] *
                            ds.config["user"]["n_constrictions"])
    assert np.allclose(area_of_region1, area_of_region1_calc)


def test_pf_with_user_config_section_fails():
    """Use a plugin feature with the user defined config section"""
    info = {"method": compute_with_user_section,
            "feature names": ["area_of_region"],
            "config required": [["user", ["n_constrictions"]]]}
    PlugInFeature("area_of_region", info)

    ds = dclab.new_dataset(retrieve_data("fmt-hdf5_fl_2018.zip"))
    # show that the plugin feature is not available before setting the
    # user metadata
    ds.config["user"].clear()
    with pytest.raises(KeyError,
                       match=r"Feature \'area_of_region\' does not exist"):
        ds["area_of_region"]
    # show that the plugin fails when the user metadata type is wrong
    ds.config["user"]["n_constrictions"] = 4.99
    with pytest.raises(AssertionError, match="should be an integer value"):
        ds["area_of_region"]


def test_pf_wrong_data_shape_1():
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    with dclab.new_dataset(h5path) as ds:
        info = example_plugin_info_single_feature()
        info["scalar feature"] = [False]
        pf = PlugInFeature("circ_per_area", info)
        with pytest.raises(ValueError, match="is not a scalar feature"):
            ds[pf.feature_name]


def test_pf_wrong_data_shape_2():
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    with dclab.new_dataset(h5path) as ds:
        info = example_plugin_info_single_feature()
        info["scalar feature"] = [True]
        info["method"] = lambda x: np.arange(len(ds) * 2).reshape(-1, 2)
        pf = PlugInFeature("circ_per_area", info)
        with pytest.raises(ValueError, match="is a scalar feature"):
            ds[pf.feature_name]


def test_pf_wrong_length_1():
    """plugin feature should have same length"""
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    with dclab.new_dataset(h5path) as ds:
        info = example_plugin_info_single_feature()
        info["method"] = lambda x: np.arange(len(ds) // 2)
        pf = PlugInFeature("circ_per_area", info)
        with pytest.warns(BadFeatureSizeWarning,
                          match="to match event number"):
            ds[pf.feature_name]


def test_pf_wrong_length_2():
    """plugin feature should have same length"""
    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    with dclab.new_dataset(h5path) as ds:
        info = example_plugin_info_single_feature()
        info["method"] = lambda x: np.arange(len(ds) * 2)
        pf = PlugInFeature("circ_per_area", info)
        with pytest.warns(BadFeatureSizeWarning,
                          match="to match event number"):
            ds[pf.feature_name]


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
            remove_all_plugin_features()
