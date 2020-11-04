"""ML: Using RT-DC data with tensorflow

We use tensorflow to distinguish between beads and cells using
scalar features only. The example data is taken from a `reference
dataset on DCOR <https://dcor.mpl.mpg.de/dataset/figshare-7771184-v2>`_.
The classification accuracy using only the inputs ``area_ratio``,
``area_um``, ``bright_sd``, and ``deform`` reaches values above 95%.

.. warning::

   This example neglects a lot of important aspects of machine learning
   with RT-DC data (e.g. brightness normalization) and it is a very
   easy task (beads are smaller than cells). Thus, this example should
   only be considered as a technical guide on how tensorflow can be used
   with RT-DC data.

.. note::

   What happens when you add ``"bright_avg"`` to the ``features`` list?
   Can you explain the result?

Apparently, debris in the cell dataset is classified as beads.
We could have gotten around that by filtering the input data before
inference. In addition, some beads get classified as cells as well.
This is a result of the limited features used for training/inference.
Under normal cirumstances, you would investigate other features in
order to improve the model prediction.
"""
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
from dclab.ml import tf_dataset

tf.random.set_seed(42)  # for reproducibility

# https://dcor.mpl.mpg.de/dataset/figshare-7771184-v2
dcor_ids = ["fb719fb2-bd9f-817a-7d70-f4002af916f0",
            "f7fa778f-6abd-1b53-ae5f-9ce12601d6f8"]
labels = [0, 1]  # 0: beads, 1: cells
features = ["area_ratio", "area_um", "bright_sd", "deform"]

# obtain train and test datasets
train, test = tf_dataset.assemble_tf_dataset_scalars(
    dc_data=dcor_ids,  # can also be list of paths or datasets
    labels=labels,
    feature_inputs=features,
    split=.8)

# build the model
model = tf.keras.Sequential(
    layers=[
        tf.keras.layers.Input(shape=(len(features),)),
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
model.fit(train, epochs=5)

# show accuracy using test data (loss: 0.1139 - accuracy: 0.9659)
model.evaluate(test, verbose=2)

# predict classes of the test data
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
y_test = np.concatenate([y for x, y in test], axis=0)
predict = np.argmax(probability_model.predict(test), axis=1)

# take a few exemplary events from true and false classification
false_cl = np.where(predict != y_test)[0]
true_cl = np.where(predict == y_test)[0]
num_events = min(4, min(len(true_cl), len(false_cl)))

false_images = tf_dataset.get_dataset_event_feature(
    dc_data=dcor_ids,
    feature="image",
    tf_dataset_indices=false_cl[:num_events],
    split_index=1,
    split=.8)

true_images = tf_dataset.get_dataset_event_feature(
    dc_data=dcor_ids,
    feature="image",
    tf_dataset_indices=true_cl[:num_events],
    split_index=1,
    split=.8)

fig = plt.figure(figsize=(8, 7))

for ii in range(num_events):
    title_true = ("cell" if y_test[true_cl[[ii]]] else "bead") + " (correct)"
    title_false = ("cell" if predict[false_cl[ii]] else "bead") + " (wrong)"
    ax1 = plt.subplot(num_events, 2, 2*ii+1, title=title_true)
    ax2 = plt.subplot(num_events, 2, 2*(ii + 1), title=title_false)
    ax1.axis("off")
    ax2.axis("off")
    ax1.imshow(true_images[ii], cmap="gray")
    ax2.imshow(false_images[ii], cmap="gray")

plt.tight_layout()
plt.show()
