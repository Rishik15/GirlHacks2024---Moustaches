from src.exception import CustomException
from src.logger import logging
import cv2
import imghdr
from dataclasses import dataclass
import os
import sys
import tensorflow as tf
import numpy as np


class dataIngestion:
    def __init__(self) -> None:
       pass
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            logging.info("Handling dodgy images.")
            data_dir = 'GirlHacks2024---Moustaches/data/raw'

            handleDodgyImage(data_dir = data_dir)

            data = tf.keras.utils.image_dataset_from_directory(data_dir)
            logging.info("Dataset loaded successfully.")

            data_iterator = data.as_numpy_iterator()

            batch = data_iterator.next()
            logging.info(f"Batch shape: {batch[0].shape}")

            return data


        except Exception as e:
            raise CustomException(e, sys)
        

def handleDodgyImage(data_dir: str):
    image_exts = ['jpeg', 'jpg', 'bmp', 'png']
    for image_class in os.listdir(data_dir):
        for image in os.listdir(os.path.join(data_dir, image_class)):
            image_path = os.path.join(data_dir, image_class, image)
            try:
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_exts:
                    print(f'Image not in extensions list {image_path}')
                    os.remove(image_path)
            except Exception as e:
                raise CustomException(e, sys)
