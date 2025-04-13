
import argparse
import tensorflow as tf
import keras_efficientnet_v2
import numpy as np

def get_model(input_size,
              num_class,
              dropout_rate = [0.3, 0.3],
              weight_name = 'imagenet21k',
              bottleneck_dense = 1280,
              *args,
              **kwargs):#,
              #FixRes = False):

    base_model = keras_efficientnet_v2.EfficientNetV2S(
        input_shape = (*input_size, 3),
        drop_connect_rate = dropout_rate[0],
        num_classes = 0,
        pretrained= weight_name)

    #base_model2= tf.keras.applications.resnet50.ResNet50(
    #include_top=False,
    #weights='imagenet',
    #input_shape=(*crop_resol, 3),
    #pooling=None
#)
    output =tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(dropout_rate[0]),
        tf.keras.layers.Dense(bottleneck_dense, activation="relu"),
        tf.keras.layers.Dropout(dropout_rate[1]),
        tf.keras.layers.Dense(num_class, activation="softmax")

    ])

    return output
