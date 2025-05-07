import tensorflow as tf
from tensorflow.keras import layers # Changed import
from tensorflow.keras import models # Changed import

class AI85Net5(tf.keras.Model):
    def __init__(self, num_classes=10, num_channels=1, dimensions=(28, 28),
                 planes=60, pool=2, fc_inputs=12, **kwargs):
        super(AI85Net5, self).__init__(**kwargs)

        dim = dimensions[0]

        # Conv1: Conv + ReLU
        self.conv1 = models.Sequential([
            layers.Conv2D(filters=planes, kernel_size=3, padding='same', use_bias=False, input_shape=(dimensions[0], dimensions[1], num_channels)),
            layers.ReLU(),
            tf.keras.layers.Dropout(.2)
        ], name = "Conv2DReLU")

        # Conv2: MaxPool + Conv + ReLU
        pad = 2 if dim == 28 else 1
        self.conv2 = models.Sequential([
            layers.ZeroPadding2D(padding=pad),
            layers.MaxPooling2D(pool_size=2, strides=2),
            layers.Conv2D(filters=planes, kernel_size=3, padding='valid', use_bias=False),
            layers.ReLU(),
            tf.keras.layers.Dropout(.4)
        ], name = "MaxPoolConv2DReLU")
        dim //= 2
        if pad == 2:
            dim += 2

        # Conv3: MaxPool + Conv + ReLU
        self.conv3 = models.Sequential([
            layers.MaxPooling2D(pool_size=2, strides=2),
            layers.Conv2D(filters=128 - planes - fc_inputs, kernel_size=3, padding='same', use_bias=False),
            layers.ReLU(),
            tf.keras.layers.Dropout(.6)
        ], name = "MaxPoolConv2DReLU")
        dim //= 2

        # Conv4: AvgPool + Conv + ReLU
        self.conv4 = models.Sequential([
            layers.AveragePooling2D(pool_size=pool, strides=2),
            layers.Conv2D(filters=fc_inputs, kernel_size=3, padding='same', use_bias=False),
            layers.ReLU(),
            tf.keras.layers.Dropout(.6)
        ], name = "AvgPoolConv2DReLU")
        dim //= pool

        # Fully connected
        self.final = models.Sequential([
            layers.Flatten(),
            layers.Dense(num_classes, activation="softmax")
        ], name = "FlattenSoftmax")

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.final(x)
        return x