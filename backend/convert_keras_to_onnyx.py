# tools/convert_keras_to_onnx.py  (run in Python_env_311)
import os
import numpy as np

# NumPy 2.x alias shims (harmless if NumPy<2; keeps tf2onnx happy)
if not hasattr(np, "bool"):    np.bool    = np.bool_
if not hasattr(np, "object"):  np.object  = object
if not hasattr(np, "int"):     np.int     = int
if not hasattr(np, "float"):   np.float   = float
if not hasattr(np, "complex"): np.complex = complex

# Use the tf-keras compat layer explicitly
from tf_keras.models import load_model
import tensorflow as tf
import tf2onnx

H5_PATH = "improved_emotion_recognition_model.h5"
OUT_DIR = "export"
os.makedirs(OUT_DIR, exist_ok=True)

# Match your preprocessing: (N,48,48,1) float32 in [0,1]
spec = (tf.TensorSpec([None, 48, 48, 1], tf.float32, name="image"),)

print("Loading:", H5_PATH)
m = load_model(H5_PATH, compile=False)

print("Converting to ONNX (opset 13)…")
onnx_path = os.path.join(OUT_DIR, "emotion.onnx")
model_proto, _ = tf2onnx.convert.from_keras(
    m, input_signature=spec, opset=13, output_path=onnx_path
)
print("✅ Saved:", onnx_path)
