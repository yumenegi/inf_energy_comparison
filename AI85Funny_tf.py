import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

class AI85NASFunnyNetTFWithDropout(tf.keras.Model):
    """
    AI85NASFunnyNet Model recreated in TensorFlow/Keras with added Dropout
    """
    def __init__(
            self,
            num_classes=10,
            num_channels=3,
            dimensions=(80, 80), # Used for setting input_shape
            bias=False,
            dropout_rate_conv=0.3, # Dropout rate for convolutional blocks
            dropout_rate_dense=0.5, # Dropout rate before dense layer
            **kwargs
    ):
        super(AI85NASFunnyNetTFWithDropout, self).__init__(**kwargs)
        print(f"Creating model with num_classes={num_classes}, dimensions={dimensions}, bias={bias}")
        print(f"Dropout rates: conv={dropout_rate_conv}, dense={dropout_rate_dense}")


        # Equivalent to ai8x.FusedConv2dBNReLU(num_channels, 64, 3, stride=1, padding=1, bias=bias, batchnorm='NoAffine')
        self.conv1_1 = models.Sequential([
            layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', use_bias=bias, input_shape=(dimensions[0], dimensions[1], num_channels)),
            layers.BatchNormalization(scale=False, center=False), # Equivalent to batchnorm='NoAffine'
            layers.ReLU()
        ], name="conv1_1")

        # Equivalent to ai8x.FusedMaxPoolConv2dBNReLU(64, 32, 1, stride=1, padding=0, bias=bias, batchnorm='NoAffine')
        # MaxPool 2x2 stride 2 assumed
        self.conv1_2 = models.Sequential([
            layers.MaxPooling2D(pool_size=2, strides=2),
            layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='valid', use_bias=bias, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(scale=False, center=False),
            layers.ReLU()
        ], name="conv1_2")

        # Equivalent to ai8x.FusedConv2dBNReLU(32, 64, 3, stride=1, padding=1, bias=bias, batchnorm='NoAffine')
        self.conv1_3 = models.Sequential([
            layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', use_bias=bias, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(scale=False, center=False),
            layers.ReLU()
        ], name="conv1_3")
        

        # Equivalent to ai8x.FusedMaxPoolConv2dBNReLU(64, 32, 3, stride=1, padding=1, bias=bias, batchnorm='NoAffine')
        self.conv2_1 = models.Sequential([
            layers.MaxPooling2D(pool_size=2, strides=2),
            layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', use_bias=bias, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(scale=False, center=False),
            layers.ReLU()
        ], name="conv2_1")

        # Equivalent to ai8x.FusedConv2dBNReLU(32, 64, 1, stride=1, padding=0, bias=bias, batchnorm='NoAffine')
        self.conv2_2 = models.Sequential([
            layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='valid', use_bias=bias, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(scale=False, center=False),
            layers.ReLU()
        ], name="conv2_2")

        # Equivalent to ai8x.FusedMaxPoolConv2dBNReLU(64, 128, 3, stride=1, padding=1, bias=bias, batchnorm='NoAffine')
        self.conv3_1 = models.Sequential([
            layers.MaxPooling2D(pool_size=2, strides=2),
            layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', use_bias=bias, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(scale=False, center=False),
            layers.ReLU()
        ], name="conv3_1")

        # Equivalent to ai8x.FusedConv2dBNReLU(128, 128, 1, stride=1, padding=0, bias=bias, batchnorm='NoAffine')
        self.conv3_2 = models.Sequential([
            layers.Conv2D(filters=128, kernel_size=1, strides=1, padding='valid', use_bias=bias, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(scale=False, center=False),
            layers.ReLU()
        ], name="conv3_2")
        # Add dropout after this block
        # self.dropout3 = layers.Dropout(rate=dropout_rate_conv, name="dropout_conv3")

        # Equivalent to ai8x.FusedMaxPoolConv2dBNReLU(128, 64, 3, stride=1, padding=1, bias=bias, batchnorm='NoAffine')
        self.conv4_1 = models.Sequential([
            layers.MaxPooling2D(pool_size=2, strides=2),
            layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', use_bias=bias, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(scale=False, center=False),
            layers.ReLU()
        ], name="conv4_1")

        # Equivalent to ai8x.FusedConv2dBNReLU(64, 128, 3, stride=1, padding=1, bias=bias, batchnorm='NoAffine')
        self.conv4_2 = models.Sequential([
            layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', use_bias=bias, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(scale=False, center=False),
            layers.ReLU()
        ], name="conv4_2")
        # Add dropout after this block
        # self.dropout4 = layers.Dropout(rate=dropout_rate_conv, name="dropout_conv4")
        # Equivalent to ai8x.FusedMaxPoolConv2dBNReLU(128, 128, 1, stride=1, padding=0, bias=bias, batchnorm='NoAffine')
        # MaxPooling omitted to match 512 flat input for Dense layer
        self.conv5_1 = models.Sequential([
            layers.Conv2D(filters=128, kernel_size=1, strides=1, padding='valid', use_bias=bias, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(scale=False, center=False),
            layers.ReLU()
        ], name="conv5_1")
        # Add dropout after this block
        # self.dropout5 = layers.Dropout(rate=dropout_rate_conv, name="dropout_conv5")

        # Flatten layer equivalent to x.view(x.size(0), -1)
        self.flatten = layers.Flatten(name="flatten")
        # Add dropout before the dense layer
        # self.dropout_fc = layers.Dropout(rate=dropout_rate_dense, name="dropout_fc")


        # Equivalent to ai8x.Linear(512, num_classes, bias=bias)
        self.fc = layers.Dense(units=num_classes, use_bias=bias, activation='softmax', name="fc")

    def call(self, x, training=False):
        """Forward prop"""
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        # x = self.dropout3(x, training=training) # Apply dropout

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        # x = self.dropout4(x, training=training) # Apply dropout

        x = self.conv5_1(x)
        # x = self.dropout5(x, training=training) # Apply dropout

        x = self.flatten(x)
        # x = self.dropout_fc(x, training=training) # Apply dropout before Dense

        x = self.fc(x)
        return x

# Example usage:
# model = AI85NASFunnyNetTFWithDropout(num_classes=10, num_channels=3, dimensions=(32, 32), bias=False, dropout_rate_conv=0.3, dropout_rate_dense=0.5)
# model.build((None, 32, 32, 3)) # Build the model
# model.summary()

# To train:
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), # Or use SparseCategoricalCrossentropy if labels are integers
#               metrics=['accuracy'])
# model.fit(train_dataset, epochs=...)