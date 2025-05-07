import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

import struct

saved_model_dir = "funnyimagenet"
tflite_model_path = "converted_model.tflite"

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)