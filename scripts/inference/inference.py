import warnings
warnings.filterwarnings('ignore')
import json
import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, "../../slim/")  
from nets import nets_factory
from preprocessing import preprocessing_factory

MODEL_DIR = "/home/tom-maverick/Documents/02_GitHub/cvpr18-inaturalist-transfer/checkpoints/haute_garonne/"
IMAGE_PATH = "/home/tom-maverick/Documents/02_GitHub/cvpr18-inaturalist-transfer/data/haute_garonne/Insecta/Adalia bipunctata/1b1b9e3da1be26ca59fe42871b6d60f0.jpg"
CLASS_MANIFEST_PATH = "/home/tom-maverick/Documents/02_GitHub/cvpr18-inaturalist-transfer/data/haute_garonne/dataset_species_labels.json"
IMAGE_SIZE = 299  
NUM_CLASSES =  284
LABELS_OFFSET = 0  

with open(CLASS_MANIFEST_PATH, "r", encoding='utf-8') as file:
    species_data = json.load(file)

network_fn = nets_factory.get_network_fn(
    "inception_v3", num_classes=(NUM_CLASSES - LABELS_OFFSET), is_training=False
)

image_input = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])


logits, _ = network_fn(image_input)
predictions = tf.argmax(logits, axis=1)


variables_to_restore = tf.contrib.slim.get_variables_to_restore()
saver = tf.train.Saver(variables_to_restore)
checkpoint_path = tf.train.latest_checkpoint(MODEL_DIR)


def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert("RGB")  
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))  
    img = np.array(img, dtype=np.float32)  
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)  
    return img


with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)

    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        "inception_v3",
        is_training=False
    )

    image = preprocess_image(IMAGE_PATH)

    predicted_class = sess.run(predictions, feed_dict={image_input: image})
    if predicted_class[0] == NUM_CLASSES:
        species = f"Other: {NUM_CLASSES}"
    else:
        species = f"{species_data[str(predicted_class[0])]}: {predicted_class[0]}"
    print(predicted_class)


    img = Image.open(IMAGE_PATH)
    plt.imshow(img)
    plt.title(f"Predicted Species: {species}")
    plt.axis("off")
    plt.show()
