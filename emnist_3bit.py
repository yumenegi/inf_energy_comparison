import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import tensorflow_model_optimization as tfmot

# --- Custom 3-Bit Quantizer ---
class ThreeBitQuantizer(tfmot.quantization.keras.quantizers.Quantizer):
    def build(self, tensor_shape, name, layer):
        return {}

    def __call__(self, inputs, training, weights, **kwargs):
        return tf.quantization.fake_quant_with_min_max_vars(
            inputs, min=-1.0, max=1.0, num_bits=3
        )

    def get_config(self):
        return {}

# --- Custom QAT config using the quantizer ---
class CustomQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def get_weights_and_quantizers(self, layer):
        return [(layer.kernel, ThreeBitQuantizer())]

    def get_activations_and_quantizers(self, layer):
        return [(layer.activation, ThreeBitQuantizer())]

    def set_quantize_weights(self, layer, quantize_weights):
        layer.kernel = quantize_weights[0]

    def set_quantize_activations(self, layer, quantize_activations):
        layer.activation = quantize_activations[0]

    def get_output_quantizers(self, layer):
        return [ThreeBitQuantizer()]

    def get_config(self):
        return {}

# --- 1. Configuration ---
BATCH_SIZE = 256
EPOCHS = 15
BUFFER_SIZE = 10000

# --- 2. Load EMNIST/MNIST ---
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

num_classes = ds_info.features['label'].num_classes

# --- 3. Preprocessing ---
def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=-1)
    label = tf.one_hot(label, depth=num_classes)
    return image, label

ds_train = ds_train.map(normalize_img).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(normalize_img).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

# --- 4. Annotate layers with custom quantization ---
quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    quantize_annotate_layer(
        tf.keras.layers.Conv2D(4, (3, 3), activation='relu', padding='valid'),
        quantize_config=CustomQuantizeConfig()
    ),
    quantize_annotate_layer(
        tf.keras.layers.Conv2D(4, (3, 3), activation='relu', padding='valid'),
        quantize_config=CustomQuantizeConfig()
    ),
    tf.keras.layers.MaxPooling2D((3, 3)),

    quantize_annotate_layer(
        tf.keras.layers.Conv2D(4, (3, 3), activation='relu', padding='same'),
        quantize_config=CustomQuantizeConfig()
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    quantize_annotate_layer(
        tf.keras.layers.Dense(num_classes, activation=None),
        quantize_config=CustomQuantizeConfig()
    ),
    tf.keras.layers.Activation('softmax')
])

# --- 5. Apply Quantization Aware Training ---
with tfmot.quantization.keras.quantize_scope({
    'ThreeBitQuantizer': ThreeBitQuantizer,
    'CustomQuantizeConfig': CustomQuantizeConfig
}):
    qat_model = tfmot.quantization.keras.quantize_apply(model)

# --- 6. Compile and Train ---
qat_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

qat_model.summary()

print("\nStarting Training...")
history = qat_model.fit(
    ds_train,
    epochs=EPOCHS,
    validation_data=ds_test,
    verbose=1
)

print("Training Finished.")

# --- 7. Evaluate ---
loss, accuracy = qat_model.evaluate(ds_test, verbose=0)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
