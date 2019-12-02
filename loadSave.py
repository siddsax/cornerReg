import tensorflow as tf
import numpy as np
import sys
from tensorflow_model_optimization.sparsity import keras as sparsity

sys.path.append('layers')
sys.path.append('utils')
sys.path.append('networks')

from coord import CoordinateChannel2D

saved_model_path = sys.argv[1]
optimize_lite_model = True
full_integer_quantization = False

# num_calibration_examples = 10 
# representative_dataset = None
lite_model_file = "./lite_model.tflite"

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]


if optimize_lite_model:
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  # if representative_dataset:
  #   converter.representative_dataset = representative_dataset
  if full_integer_quantization:
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    lite_model_file = "./lite_model_quant_uint8.tflite"
lite_model_content = converter.convert()

with open(lite_model_file, "wb") as f:
  f.write(lite_model_content)
print("Wrote %sTFLite model of %d bytes." % ("optimized " if optimize_lite_model else "", len(lite_model_content)))
