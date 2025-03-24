import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
keras = tf.keras
import tensorflow_model_optimization as tfmot
from tensorflow.keras.applications import InceptionV3
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus: 
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=3600)]
        )
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
model = keras.models.load_model("./fixed-model.h5")
# model = InceptionV3(input_shape=(299, 299, 3), include_top=True, weights=None, classes=284)

# checkpoint_path = "../../checkpoints/haute_garonne/model.ckpt"  
# checkpoint = tf.train.Checkpoint(model=model)

# checkpoint.restore(tf.train.latest_checkpoint("checkpoints/")).expect_partial()
# model.summary()
# exit()

feature_description = {
    'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
    'image/format': tf.io.FixedLenFeature((), tf.string, default_value='png'),
    'image/class/label': tf.io.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
}

def _parse_function(example_proto):
    example = tf.io.parse_single_example(example_proto, feature_description)

    # Decode image and preprocess
    image = tf.io.decode_jpeg(example["image/encoded"], channels=3)
    image = tf.image.resize(image, [299, 299])  # Resize if needed
    image = tf.cast(image, tf.float32) / 255.0  # Normalize
    
    label = tf.cast(example["image/class/label"], tf.int32)
    return image, label

batch_size = 16
train_tfrecord_pattern = "../../data/haute_garonne/train_*.tfrecord"
val_tfrecord_pattern = "../../data/haute_garonne/val_*.tfrecord"
train_dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(train_tfrecord_pattern))
val_dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(val_tfrecord_pattern))

train_dataset = train_dataset.map(_parse_function).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.map(_parse_function).batch(batch_size).prefetch(tf.data.AUTOTUNE)
num_train_samples = sum(1 for _ in tf.data.TFRecordDataset(tf.io.gfile.glob(train_tfrecord_pattern)))
num_val_samples = sum(1 for _ in tf.data.TFRecordDataset(tf.io.gfile.glob(val_tfrecord_pattern)))
# batch_size = 32  # Adjust to match your training setup
steps_per_epoch = max(1, num_train_samples // batch_size)
validation_steps = max(1, num_val_samples // batch_size)

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
pruning_params = {
    "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.50,
        final_sparsity=0.80,
        begin_step=0,
        end_step=25898,  
    )
}
model_for_pruning = prune_low_magnitude(model, **pruning_params)
model_for_pruning.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)

model_for_pruning.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=30,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[tfmot.sparsity.keras.UpdatePruningStep()],
)
model_for_pruning.summary()
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
model_for_export.save("prunned.h5")
# checkpoint_path = "prunned_tf1.ckpt"
# model_for_export.save_weights(checkpoint_path)
# model_for_pruning.save("prunned.keras")