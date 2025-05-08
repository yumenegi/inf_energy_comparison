import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt # Optional: for viewing images

print("TensorFlow version:", tf.__version__)

# --- 1. Configuration ---
BATCH_SIZE = 256
EPOCHS = 15 # Train for a few epochs for demonstration
BUFFER_SIZE = 10000 # For shuffling

# --- 2. Load EMNIST Balanced Dataset ---
# Uses tensorflow_datasets (tfds) - install if necessary: pip install tensorflow-datasets
try:
    (ds_train, ds_test), ds_info = tfds.load(
        'emnist/mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True, # Returns tuple (img, label)
        with_info=True,
    )
except Exception as e:
    print(f"Error loading dataset. Make sure you have `tensorflow-datasets` installed.")
    print(f"You might need internet connection for the first download.")
    print(f"Error details: {e}")
    exit()

# Get dataset info
num_classes = ds_info.features['label'].num_classes
num_train_examples = ds_info.splits['train'].num_examples
num_test_examples = ds_info.splits['test'].num_examples

print(f"Dataset: EMNIST Letters")
print(f"Number of classes: {num_classes}")
print(f"Number of training examples: {num_train_examples}")
print(f"Number of test examples: {num_test_examples}")
print(f"Image shape: {ds_info.features['image'].shape}")

# --- 3. Preprocessing Function ---
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`, scales to [0,1], adds channel dim."""
  image = tf.cast(image, tf.float32) / 255.0
  # CNNs expect channels dimension. EMNIST is grayscale, so add last dim.
  image = tf.expand_dims(image, axis=-1)
  # One-hot encode the labels
  label = tf.one_hot(label, depth=num_classes)
  return image, label

# --- 4. Prepare Data Pipelines ---
# Apply preprocessing, shuffling, batching, and prefetching

# Training pipeline
ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache() # Cache after mapping for efficiency
ds_train = ds_train.shuffle(BUFFER_SIZE)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE) # Overlap data preprocessing and model execution

# Test pipeline
ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.cache() # Cache test data
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

# Optional: Verify a batch shape
print("\nVerifying data batch shapes:")
for images, labels in ds_train.take(1):
    print("Images batch shape:", images.shape) # (BATCH_SIZE, 28, 28, 1)
    print("Labels batch shape:", labels.shape) # (BATCH_SIZE, num_classes)


# --- 5. Build the Simple CNN Model ---
# Standard Layout: Conv -> Pool -> Conv -> Pool -> Flatten -> Dense -> Output
# Minimal Parameters: Few filters, small dense layer
input_shape = (28, 28, 1) # Height, Width, Channels

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=input_shape), # Define input shape explicitly

    # First Convolutional Block
    tf.keras.layers.Conv2D(4, (3, 3), activation='relu', padding='valid'), # 16 filters, 3x3 kernel    
    tf.keras.layers.Conv2D(4, (3, 3), activation='relu', padding='valid'), # depth_multiplier=1) # 32 filters, 3x3 kernel
    #tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((3, 3)),

    tf.keras.layers.Conv2D(4, (3, 3), activation='relu', padding='same'), # depth_multiplier=1) # 32 filters, 3x3 kernel
    #tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'), # depth_multiplier=1) # 32 filters, 3x3 kernel
    #tf.keras.layers.BatchNormalization(),    
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Flatten and Dense Layers
    tf.keras.layers.Flatten(),
    #tf.keras.layers.Dropout(0.2),
    #tf.keras.layers.Dense(72, activation='softmax'),
    #tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(num_classes, activation='softmax') # Output layer
])

# --- 6. Compile the Model ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
    loss='categorical_crossentropy', # Use categorical for one-hot labels
    metrics=['accuracy']
)

# Print model summary to see layers and parameter count
model.summary()

# --- 7. Train the Model ---
print("\nStarting Training...")
history = model.fit(
    ds_train,
    epochs=EPOCHS,
    validation_data=ds_test,
    verbose=1 # Set to 1 or 2 for progress updates
)
print("Training Finished.")

# --- 8. Evaluate the Model ---
print("\nEvaluating Model...")
loss, accuracy = model.evaluate(ds_test, verbose=0)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

for layer in model.layers: 
    print(layer.get_config(), np.round(layer.get_weights()[0] * 4), np.round(layer.get_weights()[1] * 4))
# --- Optional: Plot training history ---
# plt.figure(figsize=(12, 4))
#
# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.title('Model Accuracy')
#
# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.title('Model Loss')
#
# plt.tight_layout()
