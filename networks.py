
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



## resnet152
##linear filters 512, 512, 512
## dropout probab: 0.0001
## pretrained true
## padding_mode border
## 224, 224, 600 (batch size)
# resize_method=ResizeMethod.PAD
# LabelSmoothingCrossEntropy
# in frozen start_lr:Floats=1e-07, end_lr:Floats=10, num_it:int=100
# uses 1cycle policy
# 5 epochs for 224 no unfrozen, 5+20 for larger size
# lr = 0.01 for 224, smaller for larger sizes

class Model():

    def __init__(self, pruning_params, params):

        self.params = params
        self.pruning_params = pruning_params

    def build(self):

        inp = tf.keras.Input(shape = (self.params.image_wh, self.params.image_wh, 3))

        if self.params.baseNet == 'mobileNetV2':
            base_net = tf.keras.applications.MobileNetV2(input_shape=(self.params.image_wh, self.params.image_wh, 3), alpha = self.params.alpha, 
                                                        include_top=False)
            base_net.trainable = self.params.Btrainable
            encoder = base_net(inp)
            if self.params.useCoordConv:
                encoder = sparsity.prune_low_magnitude(layers.Conv2D(self.params.bottleneck_size, kernel_size=1, padding='valid'), **self.pruning_params)(encoder)
            if self.params.useBatchNormalization:
                encoder = layers.BatchNormalization()(encoder)
            encoder = layers.ReLU()(encoder)
            encoder = CoordinateChannel2D()(encoder)
            coordinate_regression = layers.Dense(2, activation='sigmoid') # If our corners are in [0..1] range

            tl_regression = sparsity.prune_low_magnitude(layers.Conv2D(32, kernel_size=3, padding='valid', activation='relu'), **self.pruning_params)(encoder)
            tl_regression = layers.GlobalMaxPooling2D()(tl_regression)
            tl_regression = layers.Flatten()(tl_regression)
            tl_regression = coordinate_regression(tl_regression)

            tr_regression = sparsity.prune_low_magnitude(layers.Conv2D(32, kernel_size=3, padding='valid', activation='relu'), **self.pruning_params)(encoder)
            tr_regression = layers.GlobalMaxPooling2D()(tr_regression)
            tr_regression = layers.Flatten()(tr_regression)
            tr_regression = coordinate_regression(tr_regression)

            br_regression = sparsity.prune_low_magnitude(layers.Conv2D(32, kernel_size=3, padding='valid', activation='relu'), **self.pruning_params)(encoder)
            br_regression = layers.GlobalMaxPooling2D()(br_regression)
            br_regression = layers.Flatten()(br_regression)
            br_regression = coordinate_regression(br_regression)

            bl_regression = sparsity.prune_low_magnitude(layers.Conv2D(32, kernel_size=3, padding='valid', activation='relu'), **self.pruning_params)(encoder)
            bl_regression = layers.GlobalMaxPooling2D()(bl_regression)
            bl_regression = layers.Flatten()(bl_regression)
            bl_regression = coordinate_regression(bl_regression)

            corners = layers.Concatenate()([tl_regression, tr_regression, br_regression, bl_regression])

        elif(self.params.baseNet == 'ResNet152'):
            base_net = tf.keras.applications.ResNet152(input_shape=(self.params.image_wh, self.params.image_wh, 3), 
                                                        include_top=False)
            base_net.trainable = self.params.Btrainable
            encoder = base_net(inp)
            encoder_1 = tf.keras.layers.GlobalAveragePooling2D()(encoder)
            encoder_2 = tf.keras.layers.GlobalMaxPool2D()(encoder)
            encoder = layers.Concatenate()([encoder_1, encoder_2])
            encoder = layers.Flatten()(encoder)

            encoder = layers.BatchNormalization()(encoder)
            encoder = tf.keras.layers.Dropout(0.00005)(encoder)
            encoder = layers.Dense(512, activation='relu')(encoder)

            encoder = layers.BatchNormalization()(encoder)
            encoder = tf.keras.layers.Dropout(0.00005)(encoder)
            encoder = layers.Dense(512, activation='relu')(encoder)

            encoder = layers.BatchNormalization()(encoder)
            encoder = tf.keras.layers.Dropout(0.00005)(encoder)
            encoder = layers.Dense(512, activation='relu')(encoder)

            encoder = layers.BatchNormalization()(encoder)
            encoder = tf.keras.layers.Dropout(0.0001)(encoder)
            corners = layers.Dense(8, activation='sigmoid')(encoder)

        else:
            print("Error, model not defined")
            exit()

        model = tf.keras.Model(inp, corners)
        model.summary()

        return model
