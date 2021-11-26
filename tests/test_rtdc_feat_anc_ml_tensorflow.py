"""Test machine learning tools with tensorflow"""
import pathlib
import tempfile
from unittest import mock

import numpy as np
import pytest

import dclab
from dclab import new_dataset
from dclab.rtdc_dataset import feat_anc_ml
from dclab.rtdc_dataset.feat_anc_ml import hook_tensorflow

from helper_methods import example_data_dict, retrieve_data

tf = pytest.importorskip("tensorflow")


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
    feat_anc_ml.remove_all_ml_features()


def make_data(add_feats=None, sizes=None):
    if sizes is None:
        sizes = [100, 130]
    if add_feats is None:
        add_feats = ["area_um", "deform"]
    keys = add_feats + ["time", "frame", "fl3_width"]
    data = []
    for size in sizes:
        data.append(new_dataset(example_data_dict(size=size, keys=keys)))
    return data


def make_bare_model(ml_feats=None):
    if ml_feats is None:
        ml_feats = ["area_um", "deform"]
    dc_data = make_data(ml_feats)
    # obtain train and test datasets
    tfdata = hook_tensorflow.assemble_tf_dataset_scalars(
        dc_data=dc_data,
        labels=[0, 1],
        feature_inputs=ml_feats)
    return standard_model(tfdata)


def standard_model(tfdata, epochs=1, final_size=2):
    # build the model
    num_feats = tfdata.as_numpy_iterator().next()[0].shape[1]
    model = tf.keras.Sequential(
        layers=[
            tf.keras.layers.Input(shape=(num_feats,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(final_size)
        ],
        name="scalar_features"
    )

    # fit the model to the training data
    if final_size == 1:
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    else:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    model.fit(tfdata, epochs=epochs)

    return model


def test_af_ml_score_basic():
    """Slight modification of related test in test_ml.py using ancillaries"""
    # setup
    dict1 = example_data_dict(size=1000, keys=["area_um", "aspect", "fl1_max"])
    dict1["deform"] = np.linspace(.01, .02, 1000)
    dict2 = example_data_dict(size=1000, keys=["area_um", "aspect", "fl1_max"])
    dict2["deform"] = np.linspace(.02, .03, 1000)
    ds1 = dclab.new_dataset(dict1)
    ds2 = dclab.new_dataset(dict2)
    tfdata = hook_tensorflow.assemble_tf_dataset_scalars(
        dc_data=[ds1, ds2],
        labels=[0, 1],
        feature_inputs=["deform"])
    bare_model = standard_model(tfdata, epochs=10)
    # test dataset
    dictt = example_data_dict(size=6, keys=["area_um", "aspect", "fl1_max"])
    dictt["deform"] = np.linspace(.015, .025, 6, endpoint=True)
    dst = dclab.new_dataset(dictt)
    # DC model
    dc_model = hook_tensorflow.TensorflowModel(
        bare_model=bare_model,
        inputs=["deform"],
        outputs=["ml_score_low", "ml_score_hig"])
    # register
    feat_anc_ml.MachineLearningFeature(feature_name="ml_score_low",
                                       dc_model=dc_model)
    feat_anc_ml.MachineLearningFeature(feature_name="ml_score_hig",
                                       dc_model=dc_model)
    # get the class from the predicted values
    low = dst["ml_score_low"]
    hig = dst["ml_score_hig"]
    hl = np.concatenate((low.reshape(-1, 1), hig.reshape(-1, 1)), axis=1)
    actual = np.argmax(hl, axis=1)
    # these are the expected classes
    expected = dictt["deform"] > 0.02
    # sanity check
    assert np.sum(expected) == np.sum(~expected)
    # actual verification of test case
    assert np.all(expected == actual)


def test_af_ml_score_label_registered():
    """Test whether the correct label is returned"""
    ml_feats = ["area_um", "deform"]
    bare_model = make_bare_model(ml_feats=ml_feats)
    model = hook_tensorflow.TensorflowModel(
        bare_model=bare_model,
        inputs=ml_feats,
        outputs=["ml_score_low", "ml_score_hig"],
        info={"output labels": ["Low label", "High label"]})

    # register the features
    feat_anc_ml.MachineLearningFeature(feature_name="ml_score_low",
                                       dc_model=model)
    feat_anc_ml.MachineLearningFeature(feature_name="ml_score_hig",
                                       dc_model=model)

    # get labels
    label1 = dclab.dfn.get_feature_label("ml_score_low")
    label2 = dclab.dfn.get_feature_label("ml_score_hig")
    assert label1 == "Low label"
    assert label2 == "High label"


def test_af_ml_score_registration_sanity_checks():
    ml_feats = ["area_um", "deform"]
    dc_model = hook_tensorflow.TensorflowModel(
        bare_model=make_bare_model(),
        inputs=ml_feats,
        outputs=["ml_score_abc", "ml_score_cde"]
    )
    dictt = example_data_dict(size=6, keys=["area_um", "aspect", "fl1_max"])
    dictt["deform"] = np.linspace(.015, .025, 6, endpoint=True)
    dst = dclab.new_dataset(dictt)
    assert "ml_score_abc" not in dst, "before model registration"
    feat_anc_ml.MachineLearningFeature(feature_name="ml_score_abc",
                                       dc_model=dc_model)
    assert "ml_score_abc" in dst, "after model registration"
    feat_anc_ml.remove_all_ml_features()
    assert "ml_score_abc" not in dst, "after model unregistration"


def test_af_ml_score_register_same_output_feature_should_fail():
    ml_feats = ["area_um", "deform"]
    dc_model = hook_tensorflow.TensorflowModel(
        bare_model=make_bare_model(),
        inputs=ml_feats,
        outputs=["ml_score_abc", "ml_score_cde"]
    )
    feat_anc_ml.MachineLearningFeature(feature_name="ml_score_abc",
                                       dc_model=dc_model)

    dc_model2 = hook_tensorflow.TensorflowModel(
        bare_model=make_bare_model(),
        inputs=ml_feats,
        outputs=["ml_score_abc", "ml_score_rge"]
    )

    with pytest.raises(ValueError,
                       match="Cannot register two MachineLearningFeatures"):
        feat_anc_ml.MachineLearningFeature(feature_name="ml_score_abc",
                                           dc_model=dc_model2)


def test_assemble_tf_dataset_scalars_shuffle():
    ml_feats = ["deform", "area_um"]
    dc_data = make_data(ml_feats)
    # create a shuffled dataset
    tfdata = hook_tensorflow.assemble_tf_dataset_scalars(
        dc_data=dc_data,
        labels=[0, 1],
        feature_inputs=ml_feats,
        shuffle=True
    )
    # reproduce the shuffling
    area_um = hook_tensorflow.shuffle_array(
        np.concatenate([ds["area_um"] for ds in dc_data]))
    deform = hook_tensorflow.shuffle_array(
        np.concatenate([ds["deform"] for ds in dc_data]))
    # get the actual data
    tffeats = np.concatenate([dd for dd, ll in tfdata], axis=0)
    actual_deform = tffeats[:, 0]
    actual_area_um = tffeats[:, 1]
    assert np.all(np.array(deform, dtype=np.float32) == actual_deform)
    assert np.all(np.array(area_um, dtype=np.float32) == actual_area_um)


def test_assemble_tf_dataset_scalars_split():
    ml_feats = ["deform", "area_um"]
    dc_data = make_data(ml_feats)
    # create a shuffled dataset
    tfdata = hook_tensorflow.assemble_tf_dataset_scalars(
        dc_data=dc_data,
        labels=[0, 1],
        feature_inputs=ml_feats,
        shuffle=True
    )
    tfsplit1, tfsplit2 = hook_tensorflow.assemble_tf_dataset_scalars(
        dc_data=dc_data,
        labels=[0, 1],
        feature_inputs=ml_feats,
        split=.8,
        shuffle=True,
    )
    # reproduce the splitting
    tffeats = np.concatenate([dd for dd, ll in tfdata], axis=0)
    s1feats = np.concatenate([dd for dd, ll in tfsplit1], axis=0)
    s2feats = np.concatenate([dd for dd, ll in tfsplit2], axis=0)
    assert np.all(tffeats[:len(s1feats)] == s1feats)
    assert np.all(tffeats[len(s1feats):] == s2feats)


def test_assemble_tf_dataset_scalars_split_bad():
    ml_feats = ["deform", "area_um"]
    dc_data = make_data(ml_feats)
    try:
        hook_tensorflow.assemble_tf_dataset_scalars(
            dc_data=dc_data,
            labels=[0, 1],
            feature_inputs=ml_feats,
            split=43,
            shuffle=True)
    except ValueError:
        pass
    else:
        assert False, "Invalid split parameter"


def test_assemble_tf_dataset_scalars_non_scalar():
    ml_feats = ["deform", "area_um", "image"]
    dc_data = make_data(ml_feats)
    # create a shuffled dataset
    try:
        hook_tensorflow.assemble_tf_dataset_scalars(
            dc_data=dc_data,
            labels=[0, 1],
            feature_inputs=ml_feats)
    except ValueError:
        pass
    else:
        assert False, "'image' should not be supported for scalar dataset"


def test_basic_inference():
    # setup
    dict1 = example_data_dict(size=1000, keys=["area_um", "aspect", "fl1_max"])
    dict1["deform"] = np.linspace(.01, .02, 1000)
    dict2 = example_data_dict(size=1000, keys=["area_um", "aspect", "fl1_max"])
    dict2["deform"] = np.linspace(.02, .03, 1000)
    ds1 = new_dataset(dict1)
    ds2 = new_dataset(dict2)
    tfdata = hook_tensorflow.assemble_tf_dataset_scalars(
        dc_data=[ds1, ds2],
        labels=[0, 1],
        feature_inputs=["deform"])
    bare_model = standard_model(tfdata, epochs=10)
    # test dataset
    dictt = example_data_dict(size=6, keys=["area_um", "aspect", "fl1_max"])
    dictt["deform"] = np.linspace(.015, .025, 6, endpoint=True)
    dst = new_dataset(dictt)
    # DC model
    model = hook_tensorflow.TensorflowModel(
        bare_model=bare_model,
        inputs=["deform"],
        outputs=["ml_score_low", "ml_score_hig"])
    scores = model.predict(dst)
    # get the class from the predicted values
    low = scores["ml_score_low"]
    hig = scores["ml_score_hig"]
    hl = np.concatenate((low.reshape(-1, 1), hig.reshape(-1, 1)), axis=1)
    actual = np.argmax(hl, axis=1)
    # these are the expected classes
    expected = dictt["deform"] > 0.02
    # sanity check
    assert np.sum(expected) == np.sum(~expected)
    # actual verification of test case
    assert np.all(expected == actual)


def test_basic_inference_one_output():
    # setup
    dict1 = example_data_dict(size=1000, keys=["area_um", "aspect", "fl1_max"])
    dict1["deform"] = np.linspace(.01, .02, 1000)
    dict2 = example_data_dict(size=1000, keys=["area_um", "aspect", "fl1_max"])
    dict2["deform"] = np.linspace(.02, .03, 1000)
    ds1 = new_dataset(dict1)
    ds2 = new_dataset(dict2)
    tfdata = hook_tensorflow.assemble_tf_dataset_scalars(
        dc_data=[ds1, ds2],
        labels=[0, 1],
        feature_inputs=["deform"])
    bare_model = standard_model(tfdata, epochs=10, final_size=1)
    # test dataset
    dictt = example_data_dict(size=6, keys=["area_um", "aspect", "fl1_max"])
    dictt["deform"] = np.linspace(.015, .025, 6, endpoint=True)
    dst = new_dataset(dictt)
    # DC model
    model = hook_tensorflow.TensorflowModel(
        bare_model=bare_model,
        inputs=["deform"],
        outputs=["ml_score_loh"])
    scores = model.predict(dst)
    # get the class from the predicted values
    actual = scores["ml_score_loh"] > 0.5
    # these are the expected classes
    expected = dictt["deform"] > 0.02
    # sanity check
    assert np.sum(expected) == np.sum(~expected)
    # actual verification of test case
    assert np.all(expected == actual)


def test_get_dataset_event_feature():
    ml_feats = ["deform", "area_um"]
    dc_data = make_data(ml_feats)
    # create a shuffled dataset
    tfdata = hook_tensorflow.assemble_tf_dataset_scalars(
        dc_data=dc_data,
        labels=[0, 1],
        feature_inputs=ml_feats,
        shuffle=True
    )
    # get the actual data
    tffeats = np.concatenate([dd for dd, ll in tfdata], axis=0)
    actual_deform = tffeats[:, 0]
    actual_area_um = tffeats[:, 1]
    event_index = 42
    event_deform = hook_tensorflow.get_dataset_event_feature(
        dc_data=dc_data,
        feature="deform",
        tf_dataset_indices=[event_index],
        shuffle=True
    )[0]
    assert actual_deform[event_index] == np.float32(event_deform)
    event_area_um = hook_tensorflow.get_dataset_event_feature(
        dc_data=dc_data,
        feature="area_um",
        tf_dataset_indices=[event_index],
        shuffle=True
    )
    assert actual_area_um[event_index] == np.float32(event_area_um)


def test_get_dataset_event_feature_bad_index():
    ml_feats = ["deform", "area_um"]
    dc_data = make_data(ml_feats)
    try:
        hook_tensorflow.get_dataset_event_feature(
            dc_data=dc_data,
            feature="area_um",
            tf_dataset_indices=[10 * np.sum([len(ds) for ds in dc_data])],
            shuffle=True
        )
    except IndexError:
        pass
    else:
        assert False, "Event index too large"


def test_get_dataset_event_feature_no_tf_indices():
    ml_feats = ["deform", "area_um"]
    dc_data = make_data(ml_feats)
    indices0 = hook_tensorflow.get_dataset_event_feature(
        dc_data=dc_data,
        feature="index",
        dc_data_indices=[0],
        shuffle=False
    )
    assert list(range(1, len(dc_data[0]) + 1)) == indices0

    indices1 = hook_tensorflow.get_dataset_event_feature(
        dc_data=dc_data,
        feature="index",
        dc_data_indices=[1],
        shuffle=False
    )
    assert list(range(1, len(dc_data[1]) + 1)) == indices1


def test_get_dataset_event_feature_split():
    ml_feats = ["deform", "area_um"]
    dc_data = make_data(ml_feats)
    tfsplit1, tfsplit2 = hook_tensorflow.assemble_tf_dataset_scalars(
        dc_data=dc_data,
        labels=[0, 1],
        feature_inputs=ml_feats,
        split=.8,
        shuffle=True,
    )
    event_index = 5
    event_area_um = hook_tensorflow.get_dataset_event_feature(
        dc_data=dc_data,
        feature="area_um",
        tf_dataset_indices=[event_index],
        split=.8,
        split_index=0,
        shuffle=True
    )
    s1feats = np.concatenate([dd for dd, ll in tfsplit1], axis=0)
    assert s1feats[event_index, 1] == np.float32(event_area_um)
    event_area_um2 = hook_tensorflow.get_dataset_event_feature(
        dc_data=dc_data,
        feature="area_um",
        tf_dataset_indices=[event_index],
        split=.8,
        split_index=1,
        shuffle=True
    )
    s2feats = np.concatenate([dd for dd, ll in tfsplit2], axis=0)
    assert s2feats[event_index, 1] == np.float32(event_area_um2)


def test_get_dataset_event_feature_split_bad():
    ml_feats = ["deform", "area_um"]
    dc_data = make_data(ml_feats)
    try:
        hook_tensorflow.get_dataset_event_feature(
            dc_data=dc_data,
            feature="area_um",
            tf_dataset_indices=[1],
            split=-1,
            split_index=1,
            shuffle=True
        )
    except ValueError:
        pass
    else:
        assert False, "Invalid split parameter"


def test_get_dataset_event_feature_split_bad2():
    ml_feats = ["deform", "area_um"]
    dc_data = make_data(ml_feats)
    try:
        hook_tensorflow.get_dataset_event_feature(
            dc_data=dc_data,
            feature="area_um",
            tf_dataset_indices=[1],
            split=0,
            split_index=1,
            shuffle=True
        )
    except IndexError:
        pass
    else:
        assert False, "Invalid split parameter"


def test_model_tensorflow_has_sigmoid_activation_1():
    ml_feats = ["deform", "area_um"]
    stock_model = make_bare_model(ml_feats)
    bare_model = tf.keras.Sequential([stock_model, tf.keras.layers.Dense(1)])
    dc_model = hook_tensorflow.TensorflowModel(bare_model=bare_model,
                                               inputs=ml_feats,
                                               outputs=["ml_score_tst"])
    assert not dc_model.has_sigmoid_activation()


def test_model_tensorflow_has_sigmoid_activation_2():
    ml_feats = ["deform", "area_um"]
    stock_model = make_bare_model(ml_feats)
    bare_model = tf.keras.Sequential(
        [stock_model, tf.keras.layers.Dense(1, activation="sigmoid")])
    dc_model = hook_tensorflow.TensorflowModel(bare_model=bare_model,
                                               inputs=ml_feats,
                                               outputs=["ml_score_tst"])
    assert dc_model.has_sigmoid_activation()


def test_model_tensorflow_has_softmax_layer_1():
    ml_feats = ["deform", "area_um"]
    stock_model = make_bare_model(ml_feats)
    bare_model = tf.keras.Sequential(
        [stock_model, tf.keras.layers.Dense(2)])
    dc_model = hook_tensorflow.TensorflowModel(bare_model=bare_model,
                                               inputs=ml_feats,
                                               outputs=["ml_score_tst",
                                                        "ml_score_ts2"])
    assert not dc_model.has_softmax_layer()


def test_model_tensorflow_has_softmax_layer_2():
    ml_feats = ["deform", "area_um"]
    stock_model = make_bare_model(ml_feats)
    bare_model = tf.keras.Sequential(
        [stock_model, tf.keras.layers.Dense(2), tf.keras.layers.Softmax()])
    dc_model = hook_tensorflow.TensorflowModel(bare_model=bare_model,
                                               inputs=ml_feats,
                                               outputs=["ml_score_tst",
                                                        "ml_score_ts2"])
    assert dc_model.has_softmax_layer()


def test_modc_apply_model():
    feat_anc_ml.load_ml_feature(data_dir / "feat_anc_ml_tf_rbc.modc")
    path = retrieve_data("fmt-hdf5_image-mask-blood_2021.zip")
    with dclab.new_dataset(path) as ds:
        assert np.sum(ds["ml_score_rbc"] < .5) == 12
        assert np.sum(ds["ml_score_rbc"] >= .5) == 6
        assert np.allclose(ds["ml_score_rbc"][0], 0.21917653)


def test_modc_export_model_bad_path_1():
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="dclab_ml"))
    bare_model = make_bare_model()
    afile = tmpdir / "afile"
    afile.write_text("test")
    with pytest.raises(ValueError,
                       match=r"Output `path` should be a directory"):
        feat_anc_ml.modc.export_model(path=afile,
                                      model=bare_model,
                                      )


def test_modc_export_model_bad_path_2():
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="dclab_ml"))
    bare_model = make_bare_model()
    afile = tmpdir / "afile"
    afile.write_text("test")
    with pytest.raises(
            ValueError,
            match="Model output directory should be empty"):
        feat_anc_ml.modc.export_model(path=tmpdir,
                                      model=bare_model,
                                      )


def test_modc_export_model_enforce_format():
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="dclab_ml"))
    bare_model = make_bare_model()
    feat_anc_ml.modc.export_model(path=tmpdir,
                                  model=bare_model,
                                  enforce_formats=["tensorflow-SavedModel"]
                                  )


def test_modc_export_model_enforce_format_bad_format():
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="dclab_ml"))
    bare_model = make_bare_model()
    with pytest.raises(
            ValueError,
            match="Unsupported format 'library-format-does-not-exist'"):
        feat_anc_ml.modc.export_model(path=tmpdir,
                                      model=bare_model,
                                      enforce_formats=[
                                          "library-format-does-not-exist"]
                                      )


@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.feat_anc_ml.modc.'
                            + 'ModelFormatExportFailedWarning')
def test_modc_export_model_enforce_format_bad_model():
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="dclab_ml"))
    with pytest.raises(ValueError,
                       match="Expected an object of type `Trackable`"
                             "|Expected a Trackable object for export"):
        feat_anc_ml.modc.export_model(path=tmpdir,
                                      model=mock.MagicMock(),
                                      enforce_formats=["tensorflow-SavedModel"]
                                      )


def test_modc_load_basic():
    # setup
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="dclab_ml"))
    bare_model = make_bare_model()
    model = hook_tensorflow.TensorflowModel(bare_model=bare_model,
                                            inputs=["image"],
                                            outputs=["ml_score_tst"])
    pout = tmpdir / "test.modc"
    feat_anc_ml.save_modc(path=pout, dc_models=model)
    # assert
    feat_anc_ml.load_modc(path=pout)


def test_modc_load_as_ancillary_feature():
    # setup
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="dclab_ml"))
    bare_model = make_bare_model()
    model = hook_tensorflow.TensorflowModel(bare_model=bare_model,
                                            inputs=["image"],
                                            outputs=["ml_score_tst"])
    pout = tmpdir / "test.modc"
    feat_anc_ml.save_modc(path=pout, dc_models=model)
    # assert
    aml_list = dclab.load_ml_feature(pout)
    assert len(aml_list) == 1
    assert aml_list[0].feature_name == "ml_score_tst"
    assert "ml_score_tst" in dclab.MachineLearningFeature.feature_names


def test_modc_load_bad_format():
    # setup
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="dclab_ml"))
    bare_model = make_bare_model()
    model = hook_tensorflow.TensorflowModel(bare_model=bare_model,
                                            inputs=["image"],
                                            outputs=["ml_score_tst"])
    pout = tmpdir / "test.modc"
    feat_anc_ml.save_modc(path=pout, dc_models=model)
    # assert
    try:
        feat_anc_ml.load_modc(path=pout, from_format="format-does-not-exist")
    except ValueError:
        pass
    else:
        assert False, "bad format should not work"


def test_modc_save_basic():
    # setup
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="dclab_ml"))
    bare_model = make_bare_model()
    model = hook_tensorflow.TensorflowModel(bare_model=bare_model,
                                            inputs=["deform"],
                                            outputs=["ml_score_tst"])
    pout = tmpdir / "test.modc"
    # assert
    feat_anc_ml.save_modc(path=pout, dc_models=model)


def test_modc_save_load_infer():
    # setup
    dict1 = example_data_dict(size=1000, keys=["area_um", "aspect", "fl1_max"])
    dict1["deform"] = np.linspace(.01, .02, 1000)
    dict2 = example_data_dict(size=1000, keys=["area_um", "aspect", "fl1_max"])
    dict2["deform"] = np.linspace(.02, .03, 1000)
    ds1 = new_dataset(dict1)
    ds2 = new_dataset(dict2)
    tfdata = hook_tensorflow.assemble_tf_dataset_scalars(
        dc_data=[ds1, ds2],
        labels=[0, 1],
        feature_inputs=["deform"])
    bare_model = standard_model(tfdata, epochs=10)
    # DC model
    model = hook_tensorflow.TensorflowModel(
        bare_model=bare_model,
        inputs=["deform"],
        outputs=["ml_score_low", "ml_score_hig"])
    # save model
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="dclab_ml_modc"))
    pout = tmpdir / "test.modc"
    feat_anc_ml.save_modc(path=pout, dc_models=model)
    # load model
    model_loaded = feat_anc_ml.load_modc(path=pout)[0]
    # test dataset
    dictt = example_data_dict(size=6, keys=["area_um", "aspect", "fl1_max"])
    dictt["deform"] = np.linspace(.015, .025, 6, endpoint=True)
    dst = new_dataset(dictt)
    expected = model.predict(dst)
    actual = model_loaded.predict(dst)
    # assert
    assert np.all(expected["ml_score_low"] == actual["ml_score_low"])
    assert np.all(expected["ml_score_hig"] == actual["ml_score_hig"])


def test_models_get_dataset_features():
    ml_feats = ["area_um", "deform"]
    bare_model = make_bare_model(ml_feats=ml_feats)
    mod = hook_tensorflow.TensorflowModel(bare_model=bare_model,
                                          inputs=ml_feats,
                                          outputs=["ml_score_t01",
                                                   "ml_score_t01"])
    ds = make_data(add_feats=ml_feats, sizes=[42])[0]
    fdata = mod.get_dataset_features(ds, dtype=np.float32)
    assert np.all(fdata[:, 0] == np.array(ds["area_um"], dtype=np.float32))


def test_models_get_dataset_features_with_tensorlow():
    ml_feats = ["area_um", "deform"]
    bare_model = make_bare_model(ml_feats=ml_feats)
    mod = hook_tensorflow.TensorflowModel(bare_model=bare_model,
                                          inputs=ml_feats,
                                          outputs=["ml_score_t01",
                                                   "ml_score_t01"])
    ds = make_data(add_feats=ml_feats, sizes=[42])[0]
    fdata = mod.get_dataset_features(ds, dtype=np.float32)
    tfdata = hook_tensorflow.assemble_tf_dataset_scalars(
        dc_data=[ds],
        feature_inputs=ml_feats,
        shuffle=False
    )
    tf_area_um = np.concatenate([x[:, 0] for x in tfdata], axis=0)
    assert np.all(fdata[:, 0] == tf_area_um)


def test_models_prediction_runs_through():
    ml_feats = ["area_um", "deform"]
    bare_model = make_bare_model(ml_feats=ml_feats)
    mod = hook_tensorflow.TensorflowModel(bare_model=bare_model,
                                          inputs=ml_feats,
                                          outputs=["ml_score_t01",
                                                   "ml_score_t02"])
    ds = make_data(add_feats=ml_feats, sizes=[42])[0]
    out = mod.predict(ds)
    assert "ml_score_t01" in out
    assert "ml_score_t02" in out
