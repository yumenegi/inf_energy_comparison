import tflite_runtime.interpreter as tflite
from tensorflow import keras
import numpy as np
import time
import os
import struct
import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from funnyimagenetlite import FunnyImageNet


# Load the data using the function
fn = FunnyImageNet("./data")
_, _, test_images, test_labels = fn.sets()

print(f"Test images shape: {test_images.shape}, Test labels shape: {test_labels.shape}")

# reshape for conv2d input
test_images = test_images.reshape(-1, 80, 80, 3)

interpreter = tflite.Interpreter(model_path='converted_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], np.float32(test_images[0].reshape((1, 80, 80, 3))))
interpreter.invoke()

print('tflite on cpu:')
start = time.time()
correct = 0
for i, data in enumerate(test_images):
    input_data = np.float32(data)
    input_data = input_data.reshape((1, 80, 80, 3))
    input_shape = input_details[0]['shape']
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    if np.argmax(output_data) == test_labels[i]:
        correct += 1

print('\ttime elapsed: ', time.time() - start)
print('\taccuracy: ', correct / len(test_images))

interpreter = edgetpu.make_interpreter('converted_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# run once to avoid first invoke slowdown
interpreter.set_tensor(input_details[0]['index'], np.float32(test_images[0].reshape((1, 80, 80, 3))))
interpreter.invoke()

print('coral edge tpu:')
start = time.time()
correct = 0
for i, data in enumerate(test_images):
    input_data = np.float32(data)
    input_data = input_data.reshape((1, 80, 80, 3))
    input_shape = input_details[0]['shape']
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    if np.argmax(output_data) == test_labels[i]:
        correct += 1

print('\ttime elapsed: ', time.time() - start)
print('\taccuracy: ', correct / len(test_images))
