import tensorflow as tf
import time
import os
import tf2ta.utils as mod

def get_gpu_memory_usage():
    gpu_names = tf.test.gpu_device_name()
    tot_mem_usage = 0
    if type(gpu_names)==str:
        tot_mem_usage = tf.config.experimental.get_memory_info(gpu_names)['current']
    elif type(gpu_names)==list:
        for gpu in gpu_names:
            memory_usage = tf.config.experimental.get_memory_info(gpu)['current']
            tot_mem_usage += memory_usage
    return tot_mem_usage/(1024*1024)

def create_model(image_size, num_channels):
    """Creates a simple TensorFlow model.

    Args:
        image_size (tuple): The size of the input images (height, width).
        num_channels (int): The number of color channels in the images.

    Returns:
        tf.keras.Model: The TensorFlow model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(image_size[0], image_size[1], num_channels)),
        tf.keras.layers.Dense(1024, activation='relu'),  # Increased size
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    config = dict()
    config['model'] = {'input_size': image_size,
              'num_class': 1863,

    }

    model = mod.get_model(**config['model']                         )
    return model

def generate_data(batch_size, image_size, num_channels, dtype):
    """Generates dummy input data and labels.

    Args:
        batch_size (int): The batch size.
        image_size (tuple): The size of the input images (height, width).
        num_channels (int): The number of color channels in the images.
        dtype: The TensorFlow data type to use.

    Returns:
        tuple: A tuple containing the images and labels as TensorFlow tensors.
    """
    images = tf.random.uniform(
        shape=(batch_size, image_size[0], image_size[1], num_channels),
        dtype=dtype
    )
    labels = tf.random.uniform(
        shape=(batch_size,), minval=0, maxval=10, dtype=tf.int32
    )
    return images, labels

def train_step(model, images, labels, loss_fn, optimizer):
    """Performs a single training step.

    Args:
        model (tf.keras.Model): The TensorFlow model.
        images (tf.Tensor): The input images.
        labels (tf.Tensor): The labels.
        loss_fn: The loss function.
        optimizer: The optimizer.
    """
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test_batch_size(image_size=(224, 224), num_channels=3, dtype=tf.float32):
    """Tests increasing batch sizes with a simple TensorFlow model until an OOM error occurs.

    Args:
        image_size (tuple): The size of the input images (height, width).
        num_channels (int): The number of color channels in the images (e.g., 3 for RGB).
        dtype: The TensorFlow data type to use (e.g., tf.float32, tf.float16).
    """
    model = create_model(image_size, num_channels)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    batch_size = 1
    oom_occurred = False

    print(f"Testing batch sizes with image size: {image_size}, data type: {dtype}")
    #print(f"Available GPU Memory: {get_gpu_memory_usage():.2f} MB")

  
    while not oom_occurred:
        try:
            print(f"Trying batch size: {batch_size}")
            images, labels = generate_data(batch_size, image_size, num_channels, dtype)
            train_step(model, images, labels, loss_fn, optimizer)
            print(f"  Current GPU Memory Usage: {get_gpu_memory_usage():.2f} MB")
            batch_size *= 2
            tf.keras.backend.clear_session()
        except tf.errors.ResourceExhaustedError as e:
            print(f"Out of Memory Error: {e}")
            print(f"Ran out of memory at batch size: {batch_size}")
            tf.keras.backend.clear_session()
            oom_occurred = True
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            oom_occurred = True

    print(f"Testing complete. Optimal batch size: {batch_size//2}")