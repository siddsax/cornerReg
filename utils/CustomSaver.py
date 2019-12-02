import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow_model_optimization.sparsity import keras as sparsity

class CustomSaver(tf.keras.callbacks.Callback):
  def __init__(self, saveEpochs, params):
    self.saveEpochs = saveEpochs
    self.params = params

  def on_epoch_end(self, epoch, logs={}):
 
    if (epoch + 1) % self.saveEpochs == 0:  # or save after some epoch, each k-th epoch etc.
      final_model = sparsity.strip_pruning(self.model)
      if self.params.sparsity:
        final_model.save("./models/saved_model_pr_{}".format(epoch + 1))
        final_model.save_weights("./models/saved_model_pr_{}/weights.ckpt".format(epoch + 1))
      else:
        final_model.save("./models/saved_model_{}".format(epoch + 1))
        final_model.save_weights("./models/saved_model_{}/weights.ckpt".format(epoch + 1))
