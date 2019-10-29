from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Polygon
import itertools
import os 
from tensorflow.keras import layers
import sys
from iou import getIOU
from tensorflow.keras import backend as K
from coord import CoordinateChannel2D
import argparse

class CustomSaver(tf.keras.callbacks.Callback):
  def __init__(self, saveEpochs):
    self.saveEpochs = saveEpochs

  def on_epoch_end(self, epoch, logs={}):
    if epoch % self.saveEpochs == 0:  # or save after some epoch, each k-th epoch etc.
      self.model.save("./models/saved_modelPB_{}".format(epoch))

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--epochs', dest='epochs', type=int, default=25, help='number of epochs to run')
parser.add_argument('--loadModel', dest='loadModel', type=str, default='', help='if loading a preexisting model')
parser.add_argument('--datadir', dest='dataset_directory', type=str, default='./card_synthetic_dataset', help='if loading a preexisting model')
params = parser.parse_args()

dataset_directory = params.dataset_directory
df = pd.read_csv(os.path.join(dataset_directory, 'labels.csv'), header='infer')
show_n_records = 3 #@param {type:"integer"}

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
train_len = len(df) // 2
valid_len = len(df) * 3 // 4
seed = 1
batch_size = 16 #@param {type:"integer"}
batch_size_valid = 8

train_generator = datagen.flow_from_dataframe(
    dataframe=df[:train_len],
    directory=dataset_directory,
    x_col=filenames,
    y_col=labels,
    batch_size=batch_size,
    seed=seed,
    shuffle=True,
    class_mode="other",
    target_size=target_size)
valid_generator = test_datagen.flow_from_dataframe(
    dataframe=df[train_len:valid_len],
    directory=dataset_directory,
    x_col=filenames,
    y_col=labels,
    batch_size=batch_size_valid,
    seed=seed,
    shuffle=False,
    class_mode="other",
    target_size=target_size)
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

base_net = tf.keras.applications.MobileNetV2(input_shape=(image_wh, image_wh, 3), alpha = .35, 
                                               include_top=False)
base_net.trainable = True #@param {type:"boolean"}
# is_train = True #@param {type:"boolean"}

# TODO
inp = tf.keras.Input(shape = (image_wh, image_wh, 3));
encoder = base_net(inp)


#@markdown !!! Batch normalization layer is not supported by TFLite
useBatchNormalization = True #@param {type:"boolean"}
useCoordConv = True #@param {type:"boolean"}
if useCoordConv:
  print("-"*100)
  coord_conv_size = 256 #@param {type:"integer"}
  encoder = layers.Conv2D(coord_conv_size, kernel_size=1, padding='valid')(encoder)
  if useBatchNormalization:
    encoder = layers.BatchNormalization()(encoder)
  encoder = layers.ReLU()(encoder)

  encoder = CoordinateChannel2D()(encoder)

encoder = layers.Conv2D(256, kernel_size=3, padding='valid')(encoder)
if useBatchNormalization:
  encoder = layers.BatchNormalization()(encoder)
encoder = layers.ReLU()(encoder)

coordinate_regression = layers.Dense(2, activation='sigmoid') # If our corners are in [0..1] range

tl_regression = layers.Conv2D(32, kernel_size=3, padding='valid', activation='relu')(encoder)
tl_regression = layers.GlobalMaxPooling2D()(tl_regression)
tl_regression = layers.Flatten()(tl_regression)
tl_regression = coordinate_regression(tl_regression)

tr_regression = layers.Conv2D(32, kernel_size=3, padding='valid', activation='relu')(encoder)
tr_regression = layers.GlobalMaxPooling2D()(tr_regression)
tr_regression = layers.Flatten()(tr_regression)
tr_regression = coordinate_regression(tr_regression)

br_regression = layers.Conv2D(32, kernel_size=3, padding='valid', activation='relu')(encoder)
br_regression = layers.GlobalMaxPooling2D()(br_regression)
br_regression = layers.Flatten()(br_regression)
br_regression = coordinate_regression(br_regression)

bl_regression = layers.Conv2D(32, kernel_size=3, padding='valid', activation='relu')(encoder)
bl_regression = layers.GlobalMaxPooling2D()(bl_regression)
bl_regression = layers.Flatten()(bl_regression)
bl_regression = coordinate_regression(bl_regression)

corners = layers.Concatenate()([tl_regression, tr_regression, br_regression, bl_regression])

model = tf.keras.Model(inp, corners)
model.summary()

optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, 
              loss='mean_squared_error',
              metrics=['mae', 'mse'])

callbacks = [
    CustomSaver(saveEpochs = params.epochs // 5)
]

steps_per_epoch = train_generator.n // train_generator.batch_size
print('Steps per epoch: ', steps_per_epoch)
epochs = 50 #@param {type:'integer'}
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Training loop (Used a hack for calculating IOU at end of each epoch for now)
if len(params.loadModel):
  model = tf.keras.models.load_model(params.load_model)

# for epoch in range(params.epochs):

history = model.fit_generator(generator=train_generator,
                  steps_per_epoch=steps_per_epoch,
                  epochs=params.epochs,
                  callbacks=callbacks
                    )

#    # Get IOU on validation data

#    val_gen = valid_generator
#    val_gen.batch_size = 1
#    val_gen.reset()
#    val_steps = val_gen.n // val_gen.batch_size
#    val_gen.reset()
#    pred = model.predict_generator(val_gen,
#                                 steps=val_steps,
#                                 verbose=1)
#    predictions = pred
#    columns = labels
#    results = pd.DataFrame(predictions, columns=columns)
#    results["Filenames"] = valid_generator.filenames
#    ordered_cols = ["Filenames"] + columns
#    results = results[ordered_cols]#To get the same column order

#    ious = np.array([getIOU(A, B) for A, B in zip(results.values[:, 1:], df[train_len:valid_len].values[:, 1:])])
#    print(ious.mean())

#    if epoch % 5 == 0:

#      print("save model")
#      saved_model_dir = './models/saved_modelPB_' + str(epoch)
#      tf.saved_model.save(model, saved_model_dir, include_optimizer=True)

# A, B, C, D = True, True, True, True

# num_calibration_steps = 10


# # Type A is the quantization where the weights are quantized without the use of representational data 
# # which means that it will take larger space in RAM at runtime

# if A is True:
#   lite_model_file = 'type_A.tflite'
#   converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
#   converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
#   tflite_quant_model = converter.convert()

#   with open(lite_model_file, "wb") as f:
#     f.write(tflite_quant_model)
  
#   print("------ Saved Type A -------")

# # test_gen = test_generator
# # test_gen.batch_size = 1
# # test_gen.reset()
# # test_steps = test_gen.n // test_gen.batch_size
# # test_gen.reset()
# # pred = model.predict_generator(test_gen,
# #                                steps=test_steps,
# #                                verbose=1)

# # predictions = pred
# # columns = labels
# # results = pd.DataFrame(predictions, columns=columns)
# # results["Filenames"] = test_gen.filenames
# # ordered_cols = ["Filenames"] + columns
# # results = results[ordered_cols]#To get the same column order
# # results.to_csv("results.csv", index=False)

# # # Get IOU on the test data
# # ious = np.array([getIOU(A, B) for A, B in zip(results.values[:, 1:], df[valid_len:].values[:, 1:])])
# # print(ious.mean())

# # Type B quatization where a representative dataset is used, this has the advantages that the weights need not
# # be converted back into float32 at runtime hence saving memory

# if B is True:
#   lite_model_file = 'type_B.tflite'

#   representative_dataset_gen = lambda: itertools.islice(
#       ([image[None, ...]] for batch, _ in train_generator for image in batch),
#       num_calibration_steps)

#   converter = tf.lite.TFLiteConverter.from_keras_model(model)
#   converter.optimizations = [tf.lite.Optimize.DEFAULT]
#   converter.representative_dataset = representative_dataset_gen
#   tflite_quant_model = converter.convert()  

#   with open(lite_model_file, "wb") as f:
#     f.write(tflite_quant_model)

#   print("------ Saved Type B -------")
