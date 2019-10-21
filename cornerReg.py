from __future__ import absolute_import, division, print_function, unicode_literals

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

dataset_directory = './card_synthetic_dataset'
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
    shuffle=True,
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

base_net = tf.keras.applications.MobileNetV2(input_shape=(image_wh, image_wh, 3),
                                               include_top=False)
base_net.trainable = True #@param {type:"boolean"}
# is_train = True #@param {type:"boolean"}

# TODO
inp = tf.keras.Input(shape = (224, 224, 3))
encoder = base_net(inp)

# TODO: 
encoder = CoordinateChannel2D()(encoder)

encoder = layers.Conv2D(256, kernel_size=1, padding='valid')(encoder)
# encoder = layers.BatchNormalization(axis=1)(encoder)
encoder = layers.ReLU()(encoder)

encoder = layers.Conv2D(256, kernel_size=3, padding='valid')(encoder)
# encoder = layers.BatchNormalization(axis=1)(encoder)
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

steps_per_epoch = train_generator.n // train_generator.batch_size
print('Steps per epoch: ', steps_per_epoch)
# validation_steps = 3#valid_generator.n // valid_generator.batch_size
# print('Validation steps: ', validation_steps)
epochs = 10 #@param {type:'integer'}
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
for epoch in range(epochs):

  history = model.fit_generator(generator=train_generator,
                      steps_per_epoch=steps_per_epoch,
                      # validation_data=valid_generator,
                      # validation_steps=validation_steps,
                      epochs=1
                      )

  val_gen = valid_generator
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
  results["Filenames"] = valid_generator.filenames
  ordered_cols = ["Filenames"] + columns
  results = results[ordered_cols]#To get the same column order

  ious = np.array([getIOU(A, B) for A, B in zip(results.values[:, 1:], df[train_len:valid_len].values[:, 1:])])
  print(ious.mean())


saved_model_dir = './saved_modelPB'
tf.saved_model.save(model, saved_model_dir)

A, B, C, D = True, True, True, True

num_calibration_steps = 10

if A is True:
  lite_model_file = 'type_A.tflite'
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
  converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
  tflite_quant_model = converter.convert()

  with open(lite_model_file, "wb") as f:
    f.write(tflite_quant_model)

test_gen = test_generator
test_gen.batch_size = 1
test_gen.reset()
test_steps = test_gen.n // test_gen.batch_size
test_gen.reset()
pred = model.predict_generator(test_gen,
                               steps=test_steps,
                               verbose=1)

predictions = pred
columns = labels
results = pd.DataFrame(predictions, columns=columns)
results["Filenames"] = test_gen.filenames
ordered_cols = ["Filenames"] + columns
results = results[ordered_cols]#To get the same column order
results.to_csv("results.csv", index=False)

print(results[:4].values[:, 1:])
print(df[valid_len:valid_len+4].values[:, 1:])

ious = np.array([getIOU(A, B) for A, B in zip(results.values[:, 1:], df[valid_len:].values[:, 1:])])
print(ious.mean())
import pdb;pdb.set_trace()

# if B is True:
#   lite_model_file = 'type_B.tflite'
#   def representative_dataset_gen():
#     for _ in range(num_calibration_steps):
#       # Get sample input data as a numpy array in a method of your choosing.
#       yield [input]

#   converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
#   converter.optimizations = [tf.lite.Optimize.DEFAULT]
#   converter.representative_dataset = representative_dataset_gen
#   tflite_quant_model = converter.convert()  

#   with open(lite_model_file, "wb") as f:
#     f.write(tflite_quant_model)

# if C is True:

# ------------------------------------------------------

# optimize_lite_model = True  #@param {type:"boolean"}
# full_integer_quantization = False #@param {type: "boolean"}
# num_calibration_examples = 10  #@param {type:"slider", min:0, max:10, step:1}
# representative_dataset = None
# lite_model_file = "./lite_model.tflite"

# if optimize_lite_model and num_calibration_examples:

#   # representative_dataset = lambda: itertools.islice(
#   #     ([image[None, ...]] for batch, _ in train_generator for image in batch),
#   #     num_calibration_examples)
#   # lite_model_file = "./lite_model_quant.tflite"


#   def representative_dataset():
#     for _ in range(num_calibration_examples):
#       # Get sample input data as a numpy array in a method of your choosing.
#       yield [input]

#   # converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# # converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
# converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]


# if optimize_lite_model:
#   converter.optimizations = [tf.lite.Optimize.DEFAULT]
#   if representative_dataset:  # This is optional, see above.
#     converter.representative_dataset = representative_dataset
#   if full_integer_quantization:
#     converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#     converter.inference_input_type = tf.uint8
#     converter.inference_output_type = tf.uint8
#     lite_model_file = "./lite_model_quant_uint8.tflite"
# lite_model_content = converter.convert()

# with open(lite_model_file, "wb") as f:
#   f.write(lite_model_content)
# print("Wrote %sTFLite model of %d bytes." %
#       ("optimized " if optimize_lite_model else "", len(lite_model_content)))