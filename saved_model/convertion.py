import tensorflow as tf

# Load the saved model
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")

# Enable both built-in and Select TensorFlow Ops (Flex)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

# Optional: Set optimization flag (can reduce size, latency)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Optional: Ensure input/output types are float32 (standard for image models)
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

# Convert the model
tflite_model = converter.convert()

# Save the converted TFLite model
with open("model_with_flex.tflite", "wb") as f:
    f.write(tflite_model)

# Optional: print input details to validate correct shape
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
print("Model Input Shape:", input_details[0]['shape'])
