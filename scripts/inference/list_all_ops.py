import os
import tensorflow as tf

# 🔹 Path to model checkpoint
MODEL_DIR = "/home/tom-maverick/Documents/02_GitHub/cvpr18-inaturalist-transfer/checkpoints/inat2017_other/"

# 🔹 Find checkpoint files
meta_file = [f for f in os.listdir(MODEL_DIR) if f.endswith(".meta")][0]
print(meta_file)
meta_path = os.path.join(MODEL_DIR, meta_file)
checkpoint_path = tf.train.latest_checkpoint(MODEL_DIR)

# 🔹 Load model and print all operations
with tf.Session() as sess:
    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)
    saver.restore(sess, checkpoint_path)

    # Get the default graph
    graph = tf.get_default_graph()
    # tf.summary
    # graph.summary

    print("\n🔹 Listing all operations in the model:")
    for op in graph.get_operations():
        print(op.name)