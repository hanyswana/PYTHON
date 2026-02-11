import tensorflow as tf

# Check GPU availability
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# If no GPU is found, print an error message
if len(tf.config.list_physical_devices('GPU')) == 0:
    print("⚠️ No GPU detected! Training may be slow.")
else:
    print("✅ GPU is detected and ready for use! GPU :")
    print(tf.config.list_physical_devices('GPU'))  # Should list at least one GPU

print(tf.sysconfig.get_build_info())  # Check compiled CUDA version
print(tf.test.is_built_with_cuda())  # Should print: True
print(tf.config.list_physical_devices('GPU'))  # Should list GPU(s)


# import tensorflow as tf
# from tensorflow.python.client import device_lib
#
# # Check TensorFlow's GPU devices
# devices = device_lib.list_local_devices()
# print(devices)


# import os
# print(os.environ['LD_LIBRARY_PATH'])