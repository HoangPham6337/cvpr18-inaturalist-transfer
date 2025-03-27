from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
warnings.filterwarnings('ignore')
import math
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from tqdm import tqdm

sys.path.insert(0, "./slim/")
from datasets import dataset_factory_fgvc 
from nets import nets_factory
from preprocessing import preprocessing_factory
slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.ERROR)

def plot_confusion_matrix(cm, classes, normalized=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalized:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

MODEL_NAME = "inception_v3"
MODEL_PATH = "/home/tom-maverick/Documents/02_GitHub/cvpr18-inaturalist-transfer/checkpoints/inat2017/"
DATASET_NAME = "inat2017"
DATASET_DIR = "./data/"
CLASS_MANIFEST_PATH = "/home/tom-maverick/Documents/02_GitHub/cvpr18-inaturalist-transfer/data/inat2017/dataset_species_labels.json"
BATCH_SIZE = 100
IMAGE_SIZE = 299  
NUM_CLASSES =  1985
NUM_PREPROCESSING_THREADS = 8
LABELS_OFFSET = 0  
DATASET_SPLIT_NAME = "validation"
with tf.Graph().as_default():
    dataset = dataset_factory_fgvc.get_dataset(
        DATASET_NAME,
        DATASET_SPLIT_NAME,
        DATASET_DIR
    )
    print("Dataset split:", DATASET_SPLIT_NAME)
    print("Number of samples:", dataset.num_samples)
    print("Files:", dataset.data_sources)

    network_fn = nets_factory.get_network_fn(
        MODEL_NAME,
        num_classes=(dataset.num_classes - LABELS_OFFSET),
        is_training=False
    )

    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity = 2 * BATCH_SIZE,
        common_queue_min = BATCH_SIZE
    )

    image, label = provider.get(['image', 'label'])
    label -= LABELS_OFFSET

    preprocessing_name = MODEL_NAME
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False
    )

    image = image_preprocessing_fn(image, IMAGE_SIZE, IMAGE_SIZE)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=BATCH_SIZE,
        num_threads=NUM_PREPROCESSING_THREADS,
        capacity=5 * BATCH_SIZE
    )


    logits, _ = network_fn(images)
    predictions = tf.argmax(logits, axis=1)


    variables_to_restore = tf.contrib.slim.get_variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    checkpoint_path = tf.train.latest_checkpoint(MODEL_PATH)


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver.restore(sess, checkpoint_path)
        num_samples = dataset.num_samples
        num_batches = int(math.ceil(num_samples / float(BATCH_SIZE)))

        all_predictions = []
        all_labels = []

        try:
            for step in tqdm(range(num_batches), desc="Evaluation"):
                pred_batch, label_batch = sess.run([predictions, labels])
                all_predictions.extend(pred_batch)
                all_labels.extend(label_batch)
        except tf.errors.OutOfRangeError:
            print("Evaluation complete")
        finally:
            coord.request_stop()
            coord.join(threads)

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    with open(CLASS_MANIFEST_PATH, "r", encoding="utf-8") as file:
        species_data = json.load(file)

    class_names = [species_data[str(i)] for i in range(NUM_CLASSES)]
    cm = confusion_matrix(y_true=all_labels, y_pred=all_predictions)
    print(cm)
    # plot_confusion_matrix(cm=cm, classes=class_names, title="Confusion Matrix")



