# -*- coding: utf-8 -*-
# run bash setup.sh
# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring

import argparse
import tensorflow as tf
import tf2ta

def get_args_parser(explain_text='tf test batch size'):
    """
    get_args_parser
    """
    parser = argparse.ArgumentParser(explain_text,
                                     add_help=True)
    parser.add_argument('--image_size','-is', 
                help="test image size", type=int,
                required=True)

    return parser

if __name__ == '__main__':

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
            print(e)

    parser_main = get_args_parser()
    args = parser_main.parse_args(
    ) 

    IMAGE_SIZE = int(args.image_size)
    tf2ta.test_batch_size(image_size=(IMAGE_SIZE, IMAGE_SIZE), num_channels=3, dtype=tf.float32)

