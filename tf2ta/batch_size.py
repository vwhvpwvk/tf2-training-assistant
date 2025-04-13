import tensorflow as tf
import time
import os
import tf2ta.utils as models

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


def create_model(image_size, num_channels, num_class):
    """Creates a simple TensorFlow model.

    Args:
        image_size (tuple): The size of the input images (height, width).
        num_channels (int): The number of color channels in the images.

    Returns:
        tf.keras.Model: The TensorFlow model.
    """
    config = dict()
    config['model'] = {'input_size': image_size,
              'num_class': num_class,

    }

    model = models.get_model(**config['model']
                         )
    # update with other models as well.

    return model

def generate_data(batch_size, image_size, num_channels, num_class, dtype):
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
        shape=(batch_size,), minval=0, maxval=num_class, dtype=tf.int32
    )
    return images, labels

def load_dp_from_tfrec(ls_tfrec, 
    batch_size, 
    CROP_SIZE, 
    label_name,
    FRAME_SIZE = [1024, 768],
    convert_bgr_to_rgb = True):
    
    
    def bgr_to_rgb(tensor):
        out = tf.gather(tensor,
                indices = [2, 1, 0],
                axis = -1)
        return out
    
    def preprocess_data(example,
                        ds_keys,
                        convert_bgr = convert_bgr_to_rgb,
                        frame_size = FRAME_SIZE,
                        crop_size = CROP_SIZE):
        output = dict()
        for focal_key in ds_keys:
            if focal_key=='image':
                if convert_bgr==True:
                    img = bgr_to_rgb(example[focal_key])
                else:
                    img = img
                img = tf.image.resize(img, size = frame_size)
                img = tf.image.random_crop(img, size = [*crop_size, 3])
                output[focal_key] = img
            else:
                output[focal_key] = example[focal_key]

        return output
    
    train_ds = tfrec.read.ImageClassificationDataset(ls_tfrec)
    train_dp = next(iter(train_ds))
    ls_keys = train_dp.keys()
    preprocessed_ds = train_ds.map(lambda data: preprocess_data(data, 
    ds_keys = ls_keys))
    tr_ds = preprocessed_ds.batch(batch_size).repeat()
    tr_ds = tr_ds.map(lambda data: tfrec.read.get_image_and_label(data,
    LABEL_KEY = label_name) )
    tr_batch = next(iter(tr_ds))
    
    return tr_batch


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

def test_batch_size(image_size=(224, 224), 
                    num_channels=3, 
                    num_class = 10,
                    dtype=tf.float32, 
                    data_path=None,
                    label_name = 'label'):
    """Tests increasing batch sizes with a simple TensorFlow model until an OOM error occurs.

    Args:
        image_size (tuple): The size of the input images (height, width).
        num_channels (int): The number of color channels in the images (e.g., 3 for RGB).
        dtype: The TensorFlow data type to use (e.g., tf.float32, tf.float16).
    """
    model = create_model(image_size, num_channels, num_class)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    print(model.summary())

    batch_size = 1
    oom_occurred = False

    print(f"Testing batch sizes with image size: {image_size}, data type: {dtype}")
    # print(f"Available GPU Memory: {get_gpu_memory_usage():.2f} MB")

    # write this recursive??
    while not oom_occurred:
        try:
            print(f"Trying batch size: {batch_size}")
            if data_path:
                ls_tfrecs = tf.io.gfile.glob(data_path)
                print("data path provided. checking\n")
                print(ls_tfrecs[0])
                images, labels = load_dp_from_tfrec([ls_tfrecs[0]], 
                                    batch_size, 
                                    label_name = label_name,
                                    CROP_SIZE = image_size
                                    
                                    )
            else:
                print("no data path specified. Generating data...")
                images, labels = generate_data(batch_size, image_size, num_channels, num_class, dtype)
            train_step(model, images, labels, loss_fn, optimizer)
            print(f"  Current GPU Memory Usage: {get_gpu_memory_usage():.2f} MB")
            batch_size *= 2
            tf.keras.backend.clear_session()
        except tf.errors.ResourceExhaustedError as e:
            print(f"Out of Memory Error: {e}")
            print(f"Ran out of memory at batch size: {batch_size}")
            print(f"Optimal batch size: {batch_size//2}")
            tf.keras.backend.clear_session()
            oom_occurred = True
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            oom_occurred = True

    print(f"Testing complete.")

