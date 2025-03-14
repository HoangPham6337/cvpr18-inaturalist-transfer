import numpy as np
import tensorflow as tf
import os

slim = tf.contrib.slim

train_features = np.load("../feature/haute_garonne_train_balanced.npy")
train_labels = np.load("../feature/haute_garonne_train_labels_balanced.npy")
val_features = np.load("../feature/haute_garonne_val_balanced.npy")
val_labels = np.load("../feature/haute_garonne_val_labels_balanced.npy")

num_classes = len(set(train_labels))
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
val_labels = tf.keras.utils.to_categorical(val_labels, num_classes)

checkpoint_path = "../../checkpoints/inception/inception_v3_iNat_299.ckpt"
base_model = tf.keras.applications.InceptionV3(weights=None, include_top=False, pooling='avg')

if os.path.exists(checkpoint_path + ".index"):
    base_model.load_weights(checkpoint_path)
    print("Checkpoint loaded successfully.")
else:
    print("Checkpoint not found. Training from scratch.")

x = tf.keras.layers.Dense(1024, activation='relu')(base_model.output)
x = tf.keras.layers.Dense(512, activation='relu')(x)
predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_features, train_labels, epochs=30, batch_size=64,
          validation_data=(val_features, val_labels))

model.save_weights("./checkpoints/haute_garonne_finetuned.ckpt")
print("Fine-tuned model saved.")
