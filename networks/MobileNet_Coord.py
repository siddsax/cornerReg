
import tensorflow as tf
import numpy as np
import sys
from tensorflow_model_optimization.sparsity import keras as sparsity

sys.path.append('../layers')
sys.path.append('../utils')
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from coord import CoordinateChannel2D


class MobileNet_Coord():

    def __init__(self, pruning_params, params):

        self.params = params
        self.pruning_params = pruning_params

    def build(self):

        inp = tf.keras.Input(shape = (self.params.image_wh, self.params.image_wh, 3))

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
        coordinate_regression = layers.Dense(2)#, activation='sigmoid') # If our corners are in [0..1] range

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

        model = tf.keras.Model(inp, corners)
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        print(short_model_summary)
        writeFile = open(self.params.saveName, 'a+')
        writeFile.write(short_model_summary + "\n" + "*"*100)
        writeFile.close()

        return model
