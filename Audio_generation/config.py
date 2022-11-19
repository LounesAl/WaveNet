import tensorflow as tf

def configuration():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    session = tf.compat.v1.InteractiveSession(config=config)    