import numpy as np
import sys
import tensorflow as tf
import numpy as np

import os 
from sklearn.metrics import mean_squared_error
import argparse

sys.path.append('utils')

from iou import getIOU
from getData import getData

"""
This file basically loads a dataset defined in dataset_directory and a tflite model specified in model_path
and then tests the model on 'dummy' data which has the shape of the input data.

"""

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--datadir', dest='dataset_directory', type=str, default='./card_synthetic_dataset', help='if loading a preexisting model')
parser.add_argument('--modelPath', dest='model_path', type=str, default='./lite_model.tflite', help='if loading a preexisting model')
parser.add_argument('--noNormalize', dest='normalize', action='store_false',  help='normalizing or not')
parser.add_argument('--image_wh', dest='image_wh', type=int, default=224, help='input image width')

params = parser.parse_args()

dataset_directory = params.dataset_directory
model_path = params.model_path

_, test_generator, _, _ = getData(params, True)

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
