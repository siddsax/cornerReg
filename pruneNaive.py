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
from networks import *
import albumentations as albu
from dataHelpers import generator, create_transformer

class CustomSaver(tf.keras.callbacks.Callback):
  def __init__(self, saveEpochs):
    self.saveEpochs = saveEpochs

  def on_epoch_end(self, epoch, logs={}):
 
    if (epoch + 1) % self.saveEpochs == 0:  # or save after some epoch, each k-th epoch etc.
      final_model = sparsity.strip_pruning(self.model)
      if params.sparsity:
        final_model.save("./models/saved_model_pr_{}".format(epoch + 1))
        final_model.save_weights("./models/saved_model_pr_{}/weights.ckpt".format(epoch + 1))
      else:
        final_model.save("./models/saved_model_{}".format(epoch + 1))
        final_model.save_weights("./models/saved_model_{}/weights.ckpt".format(epoch + 1))

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--epochs', dest='epochs', type=int, default=20, help='number of epochs to run')
parser.add_argument('--loadModel', dest='loadModel', type=str, default='', help='if loading a preexisting model')
parser.add_argument('--datadir', dest='dataset_directory', type=str, default='./card_synthetic_dataset', help='if loading a preexisting model')
parser.add_argument('--logdir', dest='logdir', type=str, default='./logs', help='if loading a preexisting model')
parser.add_argument('--bottleneck_size', dest='bottleneck_size', type=int, default=64, help='size of bottleneck layer')
parser.add_argument('--useCoordConv', dest='useCoordConv', action='store_false', help='using coord convolutions or normal convolutions')
parser.add_argument('--useBatchNormalization', dest='useBatchNormalization', action='store_false', help='using batch norm or not')
parser.add_argument('--Btrainable', dest='Btrainable', action='store_false', help='if basenet is trainable or not')
parser.add_argument('--alpha', dest='alpha', type=int, default=1.0, help='width of mobilenet, only needed if mobilenet is used. Valid options are\
                                                                          .35, .5, 1.0, 2.0')
parser.add_argument('--sparsity', dest='sparsity', action='store_false', help='if basenet is trainable or not')
parser.add_argument('--baseNet', dest='baseNet', type=str, default='mobileNetV2', help='model type to load. Options MobileNetV2')
parser.add_argument('--albumentations', dest='albumentations', action='store_false', help='use albumentations or not')

params = parser.parse_args()

dataset_directory = params.dataset_directory

params.image_wh = 224
target_size = (params.image_wh, params.image_wh)
# valid_len = len(df) * 3 // 4
seed = 1
batch_size = 16 #@param {type:"integer"}
batch_size_valid = 8

lst = [x[0].split('/')[-1] for x in os.walk(dataset_directory)]

if 'train' in lst:
  dataset_directoryTR = dataset_directory + '/train'
  dataset_directoryTE = dataset_directory + '/test'
  
  trainDF = pd.read_csv(os.path.join(dataset_directoryTR, 'labels.csv'), header='infer')
  testDF = pd.read_csv(os.path.join(dataset_directoryTE, 'labels.csv'), header='infer')

  trainDF.drop(columns=['glare', 'fld_mask', 'punch'], inplace=True)
  testDF.drop(columns=['glare', 'fld_mask', 'punch'], inplace=True)

  train_len = len(trainDF)
  filenames = list(testDF)[0]
  labels = list(trainDF)[1:]

  steps_per_epoch = len(trainDF) // batch_size

else:
  df = pd.read_csv(os.path.join(dataset_directory, 'labels.csv'), header='infer')
  df.drop(columns=['glare'], inplace=True)

  labels = list(df)[1:]
  filenames = list(df)[0]
  train_len = len(df) // 2

  trainDF = df[:train_len]
  testDF = df[valid_len:]

  dataset_directoryTR = dataset_directory
  dataset_directoryTE = dataset_directory

  steps_per_epoch = len(df) // batch_size

if params.albumentations:
  transformer = create_transformer([
                                  albu.VerticalFlip(p=.5), 
                                  albu.HorizontalFlip(p=0.5),
                                  albu.Flip(p=0.5),
                                  albu.OneOf([albu.HueSaturationValue(p=0.5), albu.RGBShift(p=0.7)], p=1),
                                  albu.RandomBrightnessContrast(p=0.5)
                                  ])
else:
  transformer = create_transformer([])

train_generator = generator(trainDF, params.image_wh, batch_size, dataset_directoryTR, normalize = True, transformer = transformer)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(featurewise_center=False, rescale=1./255, horizontal_flip=False, vertical_flip=False)

# train_generator = datagen.flow_from_dataframe(
#     dataframe=df[:train_len],
#     directory=dataset_directory,
#     x_col=filenames,
#     y_col=labels,
#     batch_size=batch_size,
#     seed=seed,
#     shuffle=True,
#     class_mode="other",
#     target_size=target_size)

# valid_generator = test_datagen.flow_from_dataframe(
#     dataframe=df[train_len:valid_len],
#     directory=dataset_directory,
#     x_col=filenames,
#     y_col=labels,
#     batch_size=batch_size_valid,
#     seed=seed,
#     shuffle=False,
#     class_mode="other",
#     target_size=target_size)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=testDF,
    directory=dataset_directoryTE,
    x_col=filenames,
    y_col=labels,
    batch_size=1,
    seed=seed,
    shuffle=False,
    class_mode="other",
    target_size=target_size)

image_batch, label_batch = next(train_generator)
print("Image batch shape: ", image_batch.shape)
print("Label batch shape: ", label_batch.shape)


end_step = np.ceil(1.0 * train_len / batch_size).astype(np.int32) * params.epochs

if params.sparsity:
  pruning_params = {
        'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                    final_sparsity=0.90,
                                                    begin_step=end_step//4,
                                                    end_step=end_step,
                                                    frequency=100)
  }
else:
  pruning_params = {
        'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.0,
                                                    final_sparsity=0.0,
                                                    begin_step=1,
                                                    end_step=end_step,
                                                    frequency=100)
  }

modelInit = Model(pruning_params, params)
model = modelInit.build()

if len(params.loadModel):
  model.load_weights(params.loadModel)

#  = tf.keras.models.load_model()

optimizer = tf.keras.optimizers.Adam()#learning_rate=0.01)
model.compile(optimizer=optimizer, 
              loss='mean_squared_error',
              metrics=['mae', 'mse'])

callbacks = [
    sparsity.UpdatePruningStep(),
    sparsity.PruningSummaries(log_dir=params.logdir, profile_batch=0),
    CustomSaver(saveEpochs = params.epochs // 10)
]

# model.layers[1].trainable = False
history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=params.epochs,
                    callbacks=callbacks
                    )
