import tensorflow as tf
from tensorflow.keras import layers # Changed import
from tensorflow.keras import models # Changed import

class AI85Small(tf.keras.Model):
    def __init__(self, num_classes=10, input_shape=(28, 28, 1), fc_inputs=8):
        super().__init__()

        dim = input_shape[0]
        assert dim == input_shape[1], "Only square inputs are supported"

        self.conv1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(8, kernel_size=3, padding='same', use_bias=False, input_shape=input_shape),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(.2)
        ], name = "Conv2DReLU")

        # conv2: maxpool + conv + relu, padding = 2 if dim == 28 else 1
        pad = 2 if dim == 28 else 1
        self.conv2 = tf.keras.Sequential([
            tf.keras.layers.ZeroPadding2D(padding=pad),
            tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
            tf.keras.layers.Conv2D(8, kernel_size=3, padding='same', use_bias=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(.2)
        ], name = "MaxPoolConv2DReLU")
        dim = dim // 2 + (2 if pad == 2 else 0)

        # conv3: maxpool + conv + relu, padding=1, pool size = 4
        self.conv3 = tf.keras.Sequential([
            tf.keras.layers.ZeroPadding2D(padding=1),
            tf.keras.layers.MaxPooling2D(pool_size=4, strides=4),
            tf.keras.layers.Conv2D(fc_inputs, kernel_size=3, padding='same', use_bias=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(.2)
        ], name = "MaxPoolConv2DReLU")
        dim = dim // 4

        # Fully connected
        self.final = models.Sequential([
            layers.Flatten(),
            layers.Dense(num_classes, activation="softmax")
        ], name = "FlattenSoftmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final(x)
        return x
