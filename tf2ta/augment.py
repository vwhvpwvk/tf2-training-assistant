from . import tf

def random_fixed_size_crop(image, 
                           frame_size, 
                           crop_size):
    image = tf.image.resize(image,
                            frame_size)
    image = tf.image.random_flip_left_right(image)
    #image = tf.keras.layers.RandomRotation(
    #    (-1/12, 1/12),
    #    fill_mode = 'reflect',
    #    dtype = tf.float32
    #)(image) # RandomRotation doesn't work for mixed float. How to convert fp32 and
    # convert back to set mxflt? if set globally?
    # check first if global policy is different Random Rotation has issues
    image = tf.image.random_crop(
        image,
        (*crop_size, 3)
    )
    image = tf.cast(image, dtype = tf.float32)/255.

    return image

def bgr_to_rgb(image):
    image = tf.transpose( image,
                         perm = [2, 1, 0])
    return image

