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


"""
This file basically loads a dataset defined in dataset_directory and a tflite model specified in model_path
and then tests the model on 'dummy' data which has the shape of the input data.

"""

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--datadir', dest='dataset_directory', type=str, default='./card_synthetic_dataset', help='if loading a preexisting model')
parser.add_argument('--modelPath', dest='model_path', type=str, default='./card_synthetic_dataset', help='if loading a preexisting model')
params = parser.parse_args()


dataset_directory = params.dataset_directory #'./card_synthetic_dataset'
model_path = params.model_path
df = pd.read_csv(os.path.join(dataset_directory, 'labels.csv'), header='infer')
show_n_records = 3 #@param {type:"integer"}
# drop glare for corners regression only
df.drop(columns=['glare'], inplace=True)
print(df[:show_n_records])
print(df.columns)

labels = list(df)[1:]
print(labels)
filenames = list(df)[0]

horizontal_flip = False #@param {type:"boolean"}
vertical_flip = False#@param {type:"boolean"}
# TODO add augmentation params

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(featurewise_center=False, rescale=1./255, horizontal_flip=horizontal_flip, vertical_flip=vertical_flip)
# mobilenetv2 input size
image_wh = 224
target_size = (image_wh, image_wh)
train_len = len(df) // 2
valid_len = len(df) * 3 // 4
seed = 1


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)


test_generator = test_datagen.flow_from_dataframe(
    dataframe=df[valid_len:],
    directory=dataset_directory,
    x_col=filenames,
    y_col=labels,
    batch_size=1,
    seed=seed,
    shuffle=False,
    class_mode="other",
    target_size=target_size)

ious = []
results = []
labels = []
mse = []
for i in range(len(test_generator)):

    if i % 100 == 0:
      print(i)

    input_data = next(test_generator)
    # here input data is dummy dataset of same shape as input data
    data, label = input_data
    labels.append(label)
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    ious.append(getIOU(label, output_data))
    mse.append(mean_squared_error(label, output_data))

    results.append(output_data)

print("The Mean IOU is {} (Range 0-1) and the pixel-wise MSE error is {} (Range 0-8 for 8 points)".format(np.mean(ious), np.mean(mse)))
