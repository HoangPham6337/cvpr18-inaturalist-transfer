import tensorflow as tf
import pandas as pd
import numpy as np
import random
from tensorflow import keras
from sklearn.model_selection import train_test_split
import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
file_path = "../../data/haute_garonne/dataset_manifest.txt"
model = keras.models.load_model('model_weights.keras')

with open(file_path, "r") as f:
    lines = f.readlines()
    data = [line.strip().split(": ") for line in lines]

image_paths, labels = zip(*data)

labels = np.array(labels, dtype=np.int32)

combined = list(zip(image_paths, labels))
random.shuffle(combined)
image_paths, labels = zip(*combined)

train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [299, 299])
    image = image / 255.0
    return image, label

train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))

# Apply transformation
train_dataset = train_dataset.map(load_and_preprocess_image).batch(64).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.map(load_and_preprocess_image).batch(64).prefetch(tf.data.AUTOTUNE)

batch_size = 64
epochs = 30
validation_split = 0.1
num_images = sum(1 for _ in train_dataset) * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_step)
}

model_for_pruning = prune_low_magnitude(model, **pruning_params)
model_for_pruning.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
model_for_export.save("prunned.keras")