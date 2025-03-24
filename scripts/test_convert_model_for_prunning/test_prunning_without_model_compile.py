import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras 

new_model = keras.models.load_model('model_weights.keras')

def apply_pruning(model):
    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.1, final_sparsity=0.8, begin_step=0, end_step=1000
        )
    }
    
    # Apply pruning to Conv2D & Dense layers only
    def prune_layer(layer):
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
        return layer  

    pruned_model = tf.keras.models.clone_model(
        model,
        clone_function=prune_layer
    )

    return pruned_model

pruned_model = apply_pruning(new_model)
pruned_model.save("prunned.keras")
