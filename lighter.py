from onnx_tf.backend import prepare
import onnx

model = onnx.load("model_no_allowzero.onnx")
tf_rep = prepare(model)
tf_rep.export_graph("saved_model")  # Exports to a SavedModel directory
