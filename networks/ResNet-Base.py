import tensorflow as tf
import numpy as np
import os 
import sys

sys.path.append('../layers')
sys.path.append('../utils')

from tensorflow.keras import layers
from tensorflow.keras import backend as K

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


class ResNet_Base():

    def __init__(self, pruning_params, params):

        self.params = params
        self.pruning_params = pruning_params

    def build(self):

        inp = tf.keras.Input(shape = (self.params.image_wh, self.params.image_wh, 3))

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

        model = tf.keras.Model(inp, corners)
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        print(short_model_summary)
        writeFile = open(self.params.saveName, 'a+')
        writeFile.write(short_model_summary + "\n" + "*"*100)
        writeFile.close()

        return model
