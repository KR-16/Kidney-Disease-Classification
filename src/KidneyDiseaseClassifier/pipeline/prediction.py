import os

import numpy as np
import tensorflow as tf


class PredictionPipeline:
    #: class index -> human readable label
    CLASS_LABELS = {0: "Normal", 1: "Tumor"}

    def __init__(self, filename: str, model_path: str = os.path.join("artifacts", "training", "model.h5")):
        self.filename = filename
        self.model_path = model_path

    def predict(self):
        model = tf.keras.models.load_model(self.model_path)

        image = tf.keras.preprocessing.image.load_img(
            self.filename, target_size=(224, 224)
        )
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image, axis=0) / 255.0

        predictions = model.predict(image)
        result = int(np.argmax(predictions, axis=1)[0])
        prediction = self.CLASS_LABELS.get(result, str(result))

        return [{"image": prediction}]
