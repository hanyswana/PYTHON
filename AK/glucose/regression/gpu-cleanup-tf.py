import tensorflow as tf
from numba import cuda

def clear_gpu_memory():
    """
    Clears GPU memory in TensorFlow and CUDA.
    """
    print("Clearing TensorFlow GPU memory...")
    tf.keras.backend.clear_session()

    # Reset CUDA memory
    print("Clearing CUDA GPU memory...")
    try:
        cuda.select_device(0)  # Select GPU device (0 by default)
        cuda.close()  # Release GPU memory
        print("CUDA GPU memory cleared.")
    except Exception as e:
        print(f"Error clearing CUDA memory: {e}")

# Call the function
clear_gpu_memory()

