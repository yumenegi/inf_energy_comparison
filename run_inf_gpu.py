import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from funnyimagenet import FunnyImageNet
from AI85Funny_tf import AI85NASFunnyNetTFWithDropout

"""Initialize the model before inference"""
def init_model():
    print(tf.__version__)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    print("GPUs Available: ", tf.config.list_physical_devices('GPU'))

    fn = FunnyImageNet('./data')
    _, _, test_images, test_labels= fn.gpu_sets()

    test_images = test_images.reshape(-1, 80, 80, 3)
    print(test_images.shape)
    print(test_labels.shape)
    num_classes = 20  
    input_shape = (80, 80, 3)

    # Create an instance of the model
    model = AI85NASFunnyNetTFWithDropout(num_classes=num_classes, dropout_rate_conv=.4, dropout_rate_dense=.7, bias=True)

    # Build the model by passing input shape
    model.build(input_shape=(None,) + input_shape) # (None, 28, 28, 1)

    # Compile
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
                metrics=['accuracy'])

    model.load_weights("funnyimagenet")
    print("Model initialized")
    return model, test_images, test_labels

def eval(model, test_images, test_labels):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    loss, acc = model.evaluate(test_images, test_labels)
