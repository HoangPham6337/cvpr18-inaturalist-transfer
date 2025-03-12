from __future__ import absolute_import, division, print_function

import numpy as np
import os
import sys
import time
import tensorflow as tf
import pandas as pd

slim = tf.contrib.slim
sys.path.insert(0, '/slim/')
from nets import inception
from preprocessing import inception_preprocessing
DATA_DIR = "../data/haute_garonne"
FEATURE_SAVE_PATH = "output/haute_garonne_features.csv"
CHECKPOINT_PATH = "../checkpoints/inception_v3_iNat_299.ckpt"

IMAGE_SIZE = 299
MOVING_AVERAGE_DECAY = 0.9999

data_list = []

with tf.Graph().as_default():
    tf_global_step = tf.train.get_or_create_global_step()
    image_path = tf.placeholder(tf.string)

    image = tf.image.decode_jpeg(tf.read_file(image_path), channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = inception_preprocessing.preprocessing_image(image, IMAGE_SIZE, IMAGE_SIZE, is_training=False)
    images = tf.expand_dims(image, 0)

    
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        net, _ = inception.inception_v3_base(images, final_endpoint="Mixed_7c")

    net = tf.reduce_mean(net, [0, 1, 2])

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, tf_global_step)
    variables_to_restore = variable_averages.variables_to_restore()
    init_fn = slim.assign_from_checkpoint_fn(CHECKPOINT_PATH, variables_to_restore)

    
    config_sess = tf.ConfigProto(allow_soft_placement=True)
    config_sess.gpu_options.allow_growth = True


    with tf.Session(config=config_sess) as sess:
        init_fn(sess)
        start_time = time.time()

        for species_class in os.listdir(DATA_DIR):
            class_path = os.path.join(DATA_DIR, species_class)

            if os.path.isdir(class_path):
                for species in os.listdir(class_path):
                    species_path = os.path.join(class_path, species)

                    if os.path.isdir(species_path):
                        for img_file in os.listdir(class_path):
                            img_path = os.path.join(species_path, img_file)

                            try:
                                feature_vector = sess.run(net, feed_dict={image_path: img_path})

                                data_list.append({
                                    "Class": species_class,
                                    "Species": species,
                                    "Image Path": img_path,
                                    "Feature Vector": feature_vector.flatten()
                                })
                            except Exception as e:
                                print(f"Error processing {image_path}: {e}")
        df = pd.DataFrame(data_list)
        df_expanded = df.explode("Feature Vector")
        df_expanded.to_csv(FEATURE_SAVE_PATH, index=False)
        print(f"Extracted features saved to {FEATURE_SAVE_PATH}")
        print(f"Processing Time: {time.time() - start_time:.2f} seconds")