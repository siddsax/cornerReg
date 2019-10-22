from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Polygon
import itertools
import os 
import sys

# converter = tf.lite.TFLiteConverter.from_saved_model('saved_modelPB')  
# tflite_model = converter.convert()

saved_model_path = sys.argv[1]

optimize_lite_model = True  #@param {type:"boolean"}
full_integer_quantization = False #@param {type: "boolean"}

num_calibration_examples = 10 
representative_dataset = None
lite_model_file = "./lite_model.tflite"

# if optimize_lite_model and num_calibration_examples:

#   representative_dataset = lambda: itertools.islice(
#       ([image[None, ...]] for batch, _ in train_generator for image in batch),
#       num_calibration_examples)
#   lite_model_file = "./lite_model_quant.tflite"

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]


if optimize_lite_model:
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  if representative_dataset:  # This is optional, see above.
    converter.representative_dataset = representative_dataset
  if full_integer_quantization:
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    lite_model_file = "./lite_model_quant_uint8.tflite"
lite_model_content = converter.convert()

with open(lite_model_file, "wb") as f:
  f.write(lite_model_content)
print("Wrote %sTFLite model of %d bytes." %
      ("optimized " if optimize_lite_model else "", len(lite_model_content)))

# with open(lite_model_file, "wb") as f:
#   f.write(tflite_model)
