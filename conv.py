# load_weights_by_name_example.py
import tensorflow as tf

# Example: build a small sample model that uses Conv2D (single-frame input)
def build_single_frame_model(input_shape=(75,100,3), num_classes=10):
    inp = tf.keras.Input(shape=input_shape)  # remove time dimension here
    x = tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same")(inp)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out)

model = build_single_frame_model(input_shape=(75,100,3), num_classes=7)
# Load weights by name — layer names must match those in the H5
model.load_weights("model.h5", by_name=True, skip_mismatch=True)
model.summary()
# Then convert to TFLite when satisfied
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite = converter.convert()
open("model.tflite", "wb").write(tflite)