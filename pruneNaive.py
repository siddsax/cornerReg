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

class CustomSaver(tf.keras.callbacks.Callback):
  def __init__(self, saveEpochs):
    self.saveEpochs = saveEpochs

  def on_epoch_end(self, epoch, logs={}):
    if epoch % self.saveEpochs == 0:  # or save after some epoch, each k-th epoch etc.
      final_model = sparsity.strip_pruning(self.model)
      final_model.save("./models/saved_modelPB_pr_{}".format(epoch))

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


end_step = np.ceil(1.0 * train_len / batch_size).astype(np.int32) * params.epochs

pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                   final_sparsity=0.90,
                                                   begin_step=1600,
                                                   end_step=end_step,
                                                   frequency=100)
}


base_net = tf.keras.applications.MobileNetV2(input_shape=(image_wh, image_wh, 3), #alpha = .35, 
                                               include_top=False)
base_net.trainable = True #@param {type:"boolean"}
# is_train = True #@param {type:"boolean"}

# TODO
inp = tf.keras.Input(shape = (image_wh, image_wh, 3));
encoder = base_net(inp)

useBatchNormalization = True #@param {type:"boolean"}
useCoordConv = True #@param {type:"boolean"}
if useCoordConv:
  print("-"*100)
  bottleneck_size = 64 #@param {type:"integer"}
  encoder = sparsity.prune_low_magnitude(layers.Conv2D(bottleneck_size, kernel_size=1, padding='valid'))(encoder)
  if useBatchNormalization:
    encoder = layers.BatchNormalization()(encoder)
  encoder = layers.ReLU()(encoder)

  encoder = CoordinateChannel2D()(encoder)

#encoder = sparsity.prune_low_magnitude(layers.Conv2D(256, kernel_size=3, padding='valid'), **pruning_params)(encoder)
#if useBatchNormalization:
#  encoder = layers.BatchNormalization()(encoder)
#encoder = layers.ReLU()(encoder)

coordinate_regression = layers.Dense(2, activation='sigmoid') # If our corners are in [0..1] range

tl_regression = sparsity.prune_low_magnitude(layers.Conv2D(32, kernel_size=3, padding='valid', activation='relu'), **pruning_params)(encoder)
tl_regression = layers.GlobalMaxPooling2D()(tl_regression)
tl_regression = layers.Flatten()(tl_regression)
tl_regression = coordinate_regression(tl_regression)

tr_regression = sparsity.prune_low_magnitude(layers.Conv2D(32, kernel_size=3, padding='valid', activation='relu'), **pruning_params)(encoder)
tr_regression = layers.GlobalMaxPooling2D()(tr_regression)
tr_regression = layers.Flatten()(tr_regression)
tr_regression = coordinate_regression(tr_regression)

br_regression = sparsity.prune_low_magnitude(layers.Conv2D(32, kernel_size=3, padding='valid', activation='relu'), **pruning_params)(encoder)
br_regression = layers.GlobalMaxPooling2D()(br_regression)
br_regression = layers.Flatten()(br_regression)
br_regression = coordinate_regression(br_regression)

bl_regression = sparsity.prune_low_magnitude(layers.Conv2D(32, kernel_size=3, padding='valid', activation='relu'), **pruning_params)(encoder)
bl_regression = layers.GlobalMaxPooling2D()(bl_regression)
bl_regression = layers.Flatten()(bl_regression)
bl_regression = coordinate_regression(bl_regression)

corners = layers.Concatenate()([tl_regression, tr_regression, br_regression, bl_regression])

model = tf.keras.Model(inp, corners)
model.summary()

if len(params.loadModel):
    # saved_model_path = params.loadModel
    # model.load_weights(saved_model_path)
  model = tf.keras.models.load_model(params.loadModel)

optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, 
              loss='mean_squared_error',
              metrics=['mae', 'mse'])

callbacks = [
    sparsity.UpdatePruningStep(),
    sparsity.PruningSummaries(log_dir=params.logdir, profile_batch=0),
    CustomSaver(saveEpochs = params.epochs // 5)
]

steps_per_epoch = train_generator.n // train_generator.batch_size
print('Steps per epoch: ', steps_per_epoch)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=params.epochs,
                    callbacks=callbacks
                    )


# for epoch in range(params.epochs):

#     history = model.fit_generator(generator=train_generator,
#                         steps_per_epoch=steps_per_epoch,
#                         epochs=1,
#                         callbacks=callbacks
#                         )

#     if epoch % 5 == 0:

#       print("save model")
#       saved_model_dir = './models/saved_modelPB_pr_' + str(epoch) + '.h5'
#       # tf.saved_model.save(model, saved_model_dir)
#       # _, checkpoint_file = tempfile.mkstemp('.h5')
#       print('Saving pruned model to: ', saved_model_dir)
#       tf.keras.models.save_model(model, saved_model_dir, include_optimizer=True)

#       # Get IOU on validation data

#       val_gen = valid_generator
#       val_gen.batch_size = 1
#       val_gen.reset()
#       val_steps = val_gen.n // val_gen.batch_size
#       val_gen.reset()
#       pred = model.predict_generator(val_gen,
#                                     steps=val_steps,
#                                     verbose=1)
#       predictions = pred
#       columns = labels
#       results = pd.DataFrame(predictions, columns=columns)
#       results["Filenames"] = valid_generator.filenames
#       ordered_cols = ["Filenames"] + columns
#       results = results[ordered_cols]#To get the same column order

#       ious = np.array([getIOU(A, B) for A, B in zip(results.values[:, 1:], df[train_len:valid_len].values[:, 1:])])
#       print(ious.mean())

