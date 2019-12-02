from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import sys
import tensorflow as tf
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Polygon
import itertools
import os 
from coord import CoordinateChannel2D
from tensorflow.keras import layers
import os
from iou import getIOU
from sklearn.metrics import mean_squared_error
import argparse
import matplotlib.pylab as plt

"""
This file basically loads a dataset defined in dataset_directory and a tflite model specified in model_path
and then tests the model on 'dummy' data which has the shape of the input data.

"""

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--datadir', dest='dataset_directory', type=str, default='./card_synthetic_dataset', help='if loading a preexisting model')
parser.add_argument('--modelPath', dest='model_path', type=str, default='./lite_model.tflite', help='if loading a preexisting model')
parser.add_argument('--noNormalize', dest='normalize', action='store_false',  help='normalizing or not')
params = parser.parse_args()

dataset_directory = params.dataset_directory
model_path = params.model_path

image_wh = 224
target_size = (image_wh, image_wh)
train_len = len(df) // 2
valid_len = len(df) * 3 // 4
seed = 1

_, test_generator, _ = getData(params)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)


ious = []
results = []
labels = []
mse = []
rms = []
for i in range(len(test_generator)):

    if i % 100 == 0:
      print(i)

    input_data = next(test_generator)
    data, label = input_data
    labels.append(label)
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    ious.append(getIOU(label, output_data))
    mse.append(mean_squared_error(label, output_data))
    rms.append(np.max((label - output_data)**2))

    results.append(output_data)
    if i ==100:
       break

print("The Mean-IOU is {} (Range 0-1) point-wise MSE  is {} (Range 0-8 for 8 points) and max point RMS error is {} (Range 0-1)".format(np.mean(ious), np.mean(mse), np.mean(rms)))
