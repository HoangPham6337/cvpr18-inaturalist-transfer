import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

sess = tf.Session()
saver = tf.train.import_meta_graph('../checkpoints/haute_garonne/model.ckpt-2810.meta')
saver.restore(sess, '../checkpoints/haute_garonne/model.ckpt-2810')
graph = tf.get_default_graph()
