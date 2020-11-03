"""Test machine learning tools"""

import pathlib
import tempfile

import numpy as np
import tensorflow as tf

from dclab import ml, new_dataset

from helper_methods import example_data_dict


def make_data(add_feats=["area_um", "deform"], sizes=[100, 130]):
    keys = add_feats + ["time", "frame", "fl3_width"]
    data = []
    for size in sizes:
        data.append(new_dataset(example_data_dict(size=size, keys=keys)))
    return data


def make_bare_model(ml_feats=["area_um", "deform"]):
    dc_data = make_data(ml_feats)
    # obtain train and test datasets
    tfdata = ml.tf_dataset.assemble_tf_dataset_scalars(
        dc_data=dc_data,
        labels=[0, 1],
        feature_inputs=ml_feats)
    return standard_model(tfdata)


def standard_model(tfdata, epochs=1):
    # build the model
    num_feats = tfdata.as_numpy_iterator().next()[0].shape[1]
    model = tf.keras.Sequential(
        layers=[
            tf.keras.layers.Input(shape=(num_feats,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(2)
        ],
        name="scalar_features"
    )

    # fit the model to the training data
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    model.fit(tfdata, epochs=epochs)

    return model


def test_assemble_tf_dataset_scalars_shuffle():
    ml_feats = ["deform", "area_um"]
    dc_data = make_data(ml_feats)
    # create a shuffled dataset
    tfdata = ml.tf_dataset.assemble_tf_dataset_scalars(
        dc_data=dc_data,
        labels=[0, 1],
        feature_inputs=ml_feats,
        shuffle=True
    )
    # reproduce the shuffling
    area_um = ml.tf_dataset.shuffle_array(
        np.concatenate([ds["area_um"] for ds in dc_data]))
    deform = ml.tf_dataset.shuffle_array(
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
    tfdata = ml.tf_dataset.assemble_tf_dataset_scalars(
        dc_data=dc_data,
        labels=[0, 1],
        feature_inputs=ml_feats,
        shuffle=True
    )
    tfsplit1, tfsplit2 = ml.tf_dataset.assemble_tf_dataset_scalars(
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
        ml.tf_dataset.assemble_tf_dataset_scalars(
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
        ml.tf_dataset.assemble_tf_dataset_scalars(
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
    tfdata = ml.tf_dataset.assemble_tf_dataset_scalars(
        dc_data=[ds1, ds2],
        labels=[0, 1],
        feature_inputs=["deform"])
    bare_model = standard_model(tfdata, epochs=10)
    # test dataset
    dictt = example_data_dict(size=6, keys=["area_um", "aspect", "fl1_max"])
    dictt["deform"] = np.linspace(.015, .025, 6, endpoint=True)
    dst = new_dataset(dictt)
    # DC model
    model = ml.models.TensorflowModel(
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


def test_get_dataset_event_feature():
    ml_feats = ["deform", "area_um"]
    dc_data = make_data(ml_feats)
    # create a shuffled dataset
    tfdata = ml.tf_dataset.assemble_tf_dataset_scalars(
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
    event_deform = ml.tf_dataset.get_dataset_event_feature(
        dc_data=dc_data,
        feature="deform",
        dataset_indices=[event_index],
        shuffle=True
    )[0]
    assert actual_deform[event_index] == np.float32(event_deform)
    event_area_um = ml.tf_dataset.get_dataset_event_feature(
        dc_data=dc_data,
        feature="area_um",
        dataset_indices=[event_index],
        shuffle=True
    )
    assert actual_area_um[event_index] == np.float32(event_area_um)


def test_get_dataset_event_feature_bad_index():
    ml_feats = ["deform", "area_um"]
    dc_data = make_data(ml_feats)
    try:
        ml.tf_dataset.get_dataset_event_feature(
            dc_data=dc_data,
            feature="area_um",
            dataset_indices=[10 * np.sum([len(ds) for ds in dc_data])],
            shuffle=True
        )
    except IndexError:
        pass
    else:
        assert False, "Event index too large"


def test_get_dataset_event_feature_split():
    ml_feats = ["deform", "area_um"]
    dc_data = make_data(ml_feats)
    tfsplit1, tfsplit2 = ml.tf_dataset.assemble_tf_dataset_scalars(
        dc_data=dc_data,
        labels=[0, 1],
        feature_inputs=ml_feats,
        split=.8,
        shuffle=True,
    )
    event_index = 5
    event_area_um = ml.tf_dataset.get_dataset_event_feature(
        dc_data=dc_data,
        feature="area_um",
        dataset_indices=[event_index],
        split=.8,
        split_index=0,
        shuffle=True
    )
    s1feats = np.concatenate([dd for dd, ll in tfsplit1], axis=0)
    assert s1feats[event_index, 1] == np.float32(event_area_um)
    event_area_um2 = ml.tf_dataset.get_dataset_event_feature(
        dc_data=dc_data,
        feature="area_um",
        dataset_indices=[event_index],
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
        ml.tf_dataset.get_dataset_event_feature(
            dc_data=dc_data,
            feature="area_um",
            dataset_indices=[1],
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
        ml.tf_dataset.get_dataset_event_feature(
            dc_data=dc_data,
            feature="area_um",
            dataset_indices=[1],
            split=0,
            split_index=1,
            shuffle=True
        )
    except IndexError:
        pass
    else:
        assert False, "Invalid split parameter"


def test_models_get_dataset_features():
    ml_feats = ["area_um", "deform"]
    bare_model = make_bare_model(ml_feats=ml_feats)
    mod = ml.models.TensorflowModel(bare_model=bare_model,
                                    inputs=ml_feats,
                                    outputs=["ml_score_t01", "ml_score_t01"])
    ds = make_data(add_feats=ml_feats, sizes=[42])[0]
    fdata = mod.get_dataset_features(ds, dtype=np.float32)
    assert np.all(fdata[:, 0] == np.array(ds["area_um"], dtype=np.float32))


def test_models_get_dataset_features_with_tensorlow():
    ml_feats = ["area_um", "deform"]
    bare_model = make_bare_model(ml_feats=ml_feats)
    mod = ml.models.TensorflowModel(bare_model=bare_model,
                                    inputs=ml_feats,
                                    outputs=["ml_score_t01",
                                             "ml_score_t01"])
    ds = make_data(add_feats=ml_feats, sizes=[42])[0]
    fdata = mod.get_dataset_features(ds, dtype=np.float32)
    tfdata = ml.tf_dataset.assemble_tf_dataset_scalars(
        dc_data=[ds],
        feature_inputs=ml_feats,
        shuffle=False
    )
    tf_area_um = np.concatenate([x[:, 0] for x in tfdata], axis=0)
    assert np.all(fdata[:, 0] == tf_area_um)


def test_models_prediction_runs_through():
    ml_feats = ["area_um", "deform"]
    bare_model = make_bare_model(ml_feats=ml_feats)
    mod = ml.models.TensorflowModel(bare_model=bare_model,
                                    inputs=ml_feats,
                                    outputs=["ml_score_t01", "ml_score_t02"])
    ds = make_data(add_feats=ml_feats, sizes=[42])[0]
    out = mod.predict(ds)
    assert "ml_score_t01" in out
    assert "ml_score_t02" in out


def test_save_modc():
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="dclab_ml"))
    bare_model = make_bare_model()
    model = ml.models.TensorflowModel(bare_model=bare_model,
                                      inputs=["image"],
                                      outputs=["ml_score_tst"])
    pout = tmpdir / "test.modc"
    ml.save_modc(path=pout, dc_models=model)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
