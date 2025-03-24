# Branch Purpose

This branch documents the attempt to migrate a TensorFlow 1.x codebase (TF 1.15.1, Python 1.11.11) using tf.contrib.slim and custom training loops to a modern TF 2.x + Keras 2 API stack, with the goal of enabling model pruning and quantization using tensorflow_model_optimization.

# What was done

- Migrated trainiing and evaluation pipeline to TF2 idioms using `tf.slim.contrib`
- Converted the pre-trained TF1 `InceptionV3` model for use in TF2
- Integrated `tfmot.sparsity.keras.prune_low_magnitude` to experiment with model pruning

# Current status: Failed

**This branch is not usable in production. It exists for historical and debugging purposes.**

# Issues Encounterd
- Pruning breaks nested layers in InceptionV3
    Applying pruning to the full model, including Inception modules, led to structural conflicts.
    The recursive nature of Inception blocks (nested `tf.keras.Model`s) causes incorrect wrapper behavior.

- Training instability
    After pruning, training loss became unstable (exploding gradients), and validation accuracy dropped to ~0.042, compared to the original model's strong performance.

- Structural mismatch
    The old model's checkpoint weights did not cleanly load into the new pruned model, especially with batch normalization and nested layers. 

# Codebase Version

- Python: 1.11.11
- Tensorflow: 1.11 (Original), 2.15.1 (migration target)
- Keras: 2.15
- Model: InceptionV3 (trained with `tf.contrib.slim`)
- Pruning Tool: `tensorflow_model_optimization` (TFMOT)
