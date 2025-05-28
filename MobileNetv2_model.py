import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as tf_image

model = tf.keras.applications.MobileNetV2(weights="imagenet")

def is_human_image(image_path):
    try:
        img = tf_image.load_img(image_path, target_size=(224, 224))
        img_array = tf_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        # Check if "person" or "human" is among top predictions
        return any("person" in pred[1].lower() or "human" in pred[1].lower() for pred in decoded_predictions)
    except:
        return False