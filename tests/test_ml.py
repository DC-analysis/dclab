"""Test machine learning tools"""

import pathlib
import tempfile

import numpy as np
import tensorflow as tf

from dclab import ml

from helper_methods import example_data_dict


def make_data(add_feats=["area_um", "deform"]):
    keys = add_feats + ["time", "frame", "fl3_width"]
    ddict1 = example_data_dict(size=100, keys=keys)
    ddict2 = example_data_dict(size=130, keys=keys)
    return [ddict1, ddict2]


def make_model(ml_feats=["area_um", "deform"]):
    dc_data = make_data(ml_feats)
    # obtain train and test datasets
    tfdata = ml.tf_dataset.assemble_tf_dataset_scalars(
        dc_data=dc_data,
        labels=[0, 1],
        feature_inputs=ml_feats)

    # build the model
    model = tf.keras.Sequential(
        layers=[
            tf.keras.layers.Input(shape=(len(ml_feats),)),
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
    model.fit(tfdata, epochs=1)

    return model


def test_save_modc():
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="dclab_ml"))
    model = make_model()
    pout = tmpdir / "test.modc"
    ml.save_modc(path=pout,
                 models=model,
                 inputs=["image"],
                 outputs=["ml_score_tst"])


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


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
