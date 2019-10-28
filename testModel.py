from tensorflow_model_optimization.sparsity import keras as sparsity
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Polygon
import itertools
import os 
import sys
import matplotlib.pylab as plt
from matplotlib.patches import Polygon
from tensorflow.keras import layers
from iou import getIOU
from tensorflow.keras import backend as K
from coord import CoordinateChannel2D
import argparse
import tempfile

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--epochs', dest='epochs', type=int, default=20, help='number of epochs to run')
parser.add_argument('--loadModel', dest='loadModel', type=str, default='', help='if loading a preexisting model')
parser.add_argument('--datadir', dest='dataset_directory', type=str, default='./card_synthetic_dataset', help='if loading a preexisting model')
parser.add_argument('--logdir', dest='logdir', type=str, default='./logs', help='if loading a preexisting model')
params = parser.parse_args()


dataset_directory = params.dataset_directory
df = pd.read_csv(os.path.join(dataset_directory, 'labels.csv'), header='infer')
show_n_records = 3 #@param {type:"integer"}
# drop glare for corners regression only
df.drop(columns=['glare'], inplace=True)
print(df[:show_n_records])
print(df.columns)

labels = list(df)[1:]
print(labels)
filenames = list(df)[0]
print("Filenames column name:", filenames)
labels_txt = '\n'.join(labels)

with open('labels.txt', 'w') as f:
  f.write(labels_txt)

horizontal_flip = False #@param {type:"boolean"}
vertical_flip = False#@param {type:"boolean"}
# TODO add augmentation params

datagen = tf.keras.preprocessing.image.ImageDataGenerator(featurewise_center=False, rescale=1./255, horizontal_flip=horizontal_flip, vertical_flip=vertical_flip)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(featurewise_center=False, rescale=1./255, horizontal_flip=horizontal_flip, vertical_flip=vertical_flip)
# mobilenetv2 input size
image_wh = 224
target_size = (image_wh, image_wh)

valid_len = len(df) * 3 // 4
seed = 1
batch_size = 16 #@param {type:"integer"}
batch_size_valid = 8


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

image_batch, label_batch = train_generator[0]
print("Image batch shape: ", image_batch.shape)
print("Label batch shape: ", label_batch.shape)

import sys
saved_model_path = sys.argv[1]

if pruning:
  with sparsity.prune_scope():
    model = tf.keras.models.load_model(saved_model_path, custom_objects = {'CoordinateChannel2D' : CoordinateChannel2D})
else:
    model = tf.keras.models.load_model(saved_model_path)

# Get IOU on validation data

val_gen = test_generator
val_gen.batch_size = 1
val_gen.reset()
val_steps = val_gen.n // val_gen.batch_size
val_gen.reset()
pred = model.predict_generator(val_gen,
                            steps=val_steps,
                            verbose=1)
predictions = pred
columns = labels
results = pd.DataFrame(predictions, columns=columns)
results["Filenames"] = test_generator.filenames
ordered_cols = ["Filenames"] + columns
results = results[ordered_cols]#To get the same column order

ious = np.array([getIOU(A, B) for A, B in zip(results.values[:, 1:], df[train_len:valid_len].values[:, 1:])])
print(ious.mean())
