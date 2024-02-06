import numpy as np
import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("model_fixed.onnx")  # load onnx model
tf_model = prepare(onnx_model, auto_cast=True)
data=np.array(np.random.rand(7, 4), dtype=np.float32)
output = tf_model.run(data)  # run the loaded model
print(output)
