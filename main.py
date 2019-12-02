from tensorflow_model_optimization.sparsity import keras as sparsity
import tensorflow as tf
import numpy as np
import os 
import sys
from matplotlib.patches import Polygon
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import argparse
import datetime

sys.path.append('layers')
sys.path.append('utils')
sys.path.append('networks')
from CustomSaver import CustomSaver
from dataHelpers import generator, create_transformer
from getData import getData

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
parser.add_argument('--noSparsity', dest='noSparsity', action='store_false', help='if true, pruning is not done')
parser.add_argument('--baseNet', dest='baseNet', type=str, default='mobileNetV2', help='model type to load. Options MobileNetV2')
parser.add_argument('--noAlbumentations', dest='albumentations', action='store_false', help='use albumentations or not')
parser.add_argument('--noNormalize', dest='normalize', action='store_false',  help='normalizing or not')
parser.add_argument('--saveName', dest='saveName', default='',  help='place where to save model')
parser.add_argument('--image_wh', dest='image_wh', type=int, default=224, help='input image width')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='batch size')

params = parser.parse_args()

if params.baseNet == 'ResNet_Base':
  from ResNet-Base import ResNet_Base as Model
elif params.baseNet == 'MobileNetV2':
  from MobileNet-Coord import MobileNet_Coord as Model
else:
  print("Error; Model not defined");exit()

if len(params.saveName) == 0:
  params.saveName = 'model_' + datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")

writeFile = open(params.saveName, 'w')
writeFile.write(str(params) + "\n" + "*"*100)
writeFile.close()

train_generator, test_generator, end_step = getData(params)


if params.noSparsity:
  pruning_params = {
        'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.0,
                                                    final_sparsity=0.0,
                                                    begin_step=1,
                                                    end_step=end_step,
                                                    frequency=100)
  }
else:
  pruning_params = {
        'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                    final_sparsity=0.90,
                                                    begin_step=end_step//4,
                                                    end_step=end_step,
                                                    frequency=100)
  }

modelInit = Model(pruning_params, params)
model = modelInit.build()

if len(params.loadModel):
  model.load_weights(params.loadModel)

optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, 
              loss='mean_squared_error',
              metrics=['mae', 'mse'])

callbacks = [
    sparsity.UpdatePruningStep(),
    sparsity.PruningSummaries(log_dir=params.logdir, profile_batch=0),
    CustomSaver(saveEpochs = params.epochs // 10, params)
]

history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=params.epochs,
                    callbacks=callbacks
                    )
