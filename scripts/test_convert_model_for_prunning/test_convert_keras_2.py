import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
# from dataset_parser import get_split

model = InceptionV3(input_shape=(299, 299, 3), include_top=True, weights=None, classes=284)

checkpoint_path = "../../checkpoints/haute_garonne/model.ckpt"  
# checkpoint_path = "./.ckpt"  
checkpoint = tf.train.Checkpoint(model=model)

checkpoint.restore(tf.train.latest_checkpoint("checkpoints/")).expect_partial()

model.summary()
# val_dataset = get_split("val", "../../data/haute_garonne", num_samples={"train": 55250, "validation": 6139}, num_classes=286, shuffle=False)

# model.summary()
# model.save("model_1.keras")

# model.compile(
#     optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=[
#         tf.keras.metrics.SparseCategoricalAccuracy(),
#         tf.keras.metrics.Precision(),
#         tf.keras.metrics.Recall()
#     ]
# )

# result = model.evaluate(val_dataset)

# print(result)