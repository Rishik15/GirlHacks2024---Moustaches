import os
import sys
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from src.exception import CustomException
from src.logger import logging
import numpy as np

class PredictPipeline:
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        try:
            model_path = os.path.join("artifacts", "mushroom_model.h5")  
            model = load_model(model_path)
            logging.info("Model loaded successfully.")
            return model
        except Exception as e:
            raise CustomException(f"Error loading model: {e}", sys)

    def predict(self, image_path: str):
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise CustomException(f"Image not loaded from path: {image_path}", sys)
            
            logging.info(f"Image loaded successfully from {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized_image = tf.image.resize(img, (256, 256))  # Ensure image is resized properly
            logging.info("Image resized to 256x256.")

            # Normalize and expand dimensions for prediction
            img_array = np.expand_dims(resized_image / 255.0, axis=0)
            y_pred = self.model.predict(img_array)

            logging.info(f"Prediction raw output: {y_pred}")
            if y_pred >= 0.5:
                probability = y_pred * 100
                return {'Edibility': '1', 'Probability': probability[0][0]}
            else:
                probability = (1 - y_pred) * 100
                return {'Edibility': '0', 'Probability': probability[0][0]}
        except Exception as e:
            logging.error(f"Error in prediction: {e}")
            raise CustomException(e, sys)

