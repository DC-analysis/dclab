"""ML: Creating built-in models for dclab

The :ref:`tensorflow example <example_ml_tensorflow>` already
showcased a few convenience functions for machine learning
implemented in dclab. In this example, we want to go even
further and transform the predictions of an ML model into
an :ref:`ancillary feature <sec_features_ancillary>`
(which is then globally available in dclab).

A few things are different from the other example:

- We rename ``model`` to ``bare_model`` to make a clear
  distinction between the actual ML model (from tensorflow)
  and the model wrapper (see :ref:`sec_av_ml_models`).
- We turn the two-class problem into a regression problem
  for one feature only. Consequently, the loss function changes
  to "binary crossentropy" and for some inexplicable reason
  we have to train for 20 epochs instead of the previously 5
  to achieve convergence in accuracy.
- Finally, and this is the whole point of this example, we
  register the model as an ancillary feature and perform
  inference indirectly by simply accessing the
  ``ml_score_cel`` feature of the test dataset.

The plot shows the test fraction of the dataset. The x-axis is
(arbitrarily) set to area. The y-axis shows the sigmoid (dclab
automatically applies a sigmoid activation if it is not present
in the final layer; see :func:`dclab.ml.models.TensorflowModel.predict`)
of the model's output `logits
<https://developers.google.com/machine-learning/glossary/#logits>`_.

"""
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import dclab.ml

tf.random.set_seed(42)  # for reproducibility

# https://dcor.mpl.mpg.de/dataset/figshare-7771184-v2
dcor_ids = ["fb719fb2-bd9f-817a-7d70-f4002af916f0",
            "f7fa778f-6abd-1b53-ae5f-9ce12601d6f8"]
labels = [0, 1]  # 0: beads, 1: cells
features = ["area_ratio", "area_um", "bright_sd", "deform"]

tf_kw = {"dc_data": dcor_ids,
         "split": .8,
         "shuffle": True,
         }

# obtain train and test datasets
train, test = dclab.ml.tf_dataset.assemble_tf_dataset_scalars(
    labels=labels, feature_inputs=features, **tf_kw)

# build the model
bare_model = tf.keras.Sequential(
    layers=[
        tf.keras.layers.Input(shape=(len(features),)),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(32),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1)
    ],
    name="scalar_features"
)

# fit the model to the training data
# Note that we did not add a "sigmoid" activation function to the
# final layer and are training with logits here. We also don't
# have to manually add it in a later step, because dclab will
# add it automatically (if it does not exist) before prediction.
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
bare_model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
bare_model.fit(train, epochs=20)

# show accuracy using test data (loss: 0.0725 - accuracy: 0.9877)
bare_model.evaluate(test, verbose=2)

# register the ancillary feature "ml_score_cel" in dclab
dc_model = dclab.ml.models.TensorflowModel(
    bare_model=bare_model,
    inputs=features,
    outputs=["ml_score_cel"],
    output_labels=["Probability of having a cell"],
    model_name="Distinguish between cells and beads",
)
dc_model.register()

# Now we are actually done already. The only thing left to do is to
# visualize the prediction for the test-fraction of our dataset.
# This involves a bit of data shuffling (obtaining the dataset indices
# from the "index" feature (which starts at 1 and not 0) and creating
# hierarchy children after applying the corresponding manual filters)
# which is less complicated than it looks.

# create dataset hierarchy children for bead and cell test data
bead_train_indices = dclab.ml.tf_dataset.get_dataset_event_feature(
    feature="index", dc_data_indices=[0], split_index=0, **tf_kw)
ds_bead = dclab.new_dataset(dcor_ids[0])
ds_bead.filter.manual[np.array(bead_train_indices) - 1] = False
ds_bead.apply_filter()
ds_bead_test = dclab.new_dataset(ds_bead)  # hierarchy child with test fraction

cell_train_indices = dclab.ml.tf_dataset.get_dataset_event_feature(
    feature="index", dc_data_indices=[1], split_index=0, **tf_kw)
ds_cell = dclab.new_dataset(dcor_ids[1])
ds_cell.filter.manual[np.array(cell_train_indices) - 1] = False
ds_cell.apply_filter()
ds_cell_test = dclab.new_dataset(ds_cell)  # hierarchy child with test fraction

fig = plt.figure(figsize=(8, 7))
ax = plt.subplot(111)

plt.plot(ds_bead_test["area_um"], ds_bead_test["ml_score_cel"], ".",
         ms=10, alpha=.5, label="test data: beads")
plt.plot(ds_cell_test["area_um"], ds_cell_test["ml_score_cel"], ".",
         ms=10, alpha=.5, label="test data: cells")
leg = plt.legend()
for lh in leg.legendHandles:
    lh._legmarker.set_alpha(1)

ax.set_xlabel(dclab.dfn.get_feature_label("area_um"))
ax.set_ylabel(dclab.dfn.get_feature_label("ml_score_cel"))
ax.set_xlim(0, 130)

plt.tight_layout()
plt.show()
