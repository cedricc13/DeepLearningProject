import tensorflow as tf
import os
from tensorflow.keras import mixed_precision

def setup_tensorflow_gpu():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            
            print(f"{len(gpus)} GPU(s) setup")
            print(f"Mixed precision activated")
            return True
        except RuntimeError as e:
            print(f"GPU Error: {e}")
            return False
    else:
        print("No GPU found => switch to CPU")
        return False
