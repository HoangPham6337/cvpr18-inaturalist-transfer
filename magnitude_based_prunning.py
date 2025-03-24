import tensorflow as tf
import numpy as np

# checkpoint_path = tf.train.latest_checkpoint("checkpoints/haute_garonne_other")
# print("Latest checkpoint:", checkpoint_path)
checkpoint_path = "checkpoints/haute_garonne_other/model.ckpt-2810"
new_checkpoint_path = "checkpoints/haute_garonne_other/pruned_model.ckpt"
prune_threshold = 1e-3

reader = tf.train.NewCheckpointReader(checkpoint_path)
variable_map = reader.get_variable_to_shape_map()
pruned_vars = {}
# for var_name in variable_map:
#     print(var_name, variable_map[var_name])
#     tensor = reader.get_tensor(var_name)
with tf.Session() as sess:
    for var_name in variable_map:
        # print(var_name, variable_map[var_name])
        tensor = reader.get_tensor(var_name)

        if "weights" in var_name:
            print(f"Prunning {var_name}...")
            tensor[np.abs(tensor) < prune_threshold] = 0

        pruned_vars[var_name] = tf.Variable(tensor, name=var_name)

    saver = tf.train.Saver(var_list=pruned_vars)

    sess.run(tf.global_variables_initializer())
    saver.save(sess, new_checkpoint_path)

print(f"Pruned model saved at {new_checkpoint_path}")