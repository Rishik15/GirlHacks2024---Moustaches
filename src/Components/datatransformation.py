import sys
import os
from dataclasses import dataclass
import tensorflow as tf
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    train_data_path: str = os.path.join('data','processed', "train.csv")
    test_data_path: str = os.path.join('data', 'processed', "test.csv")
    val_data_path: str = os.path.join('data', 'processed', "validate.csv")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, data):
       try:
        data = data.map(lambda x, y :(x/255, y))
        logging.info('Scaled the data to be between 0.0 and 1.0')

        logging.info('Initializing the train, test and validation data size.')
        total_size = len(data)
        train_size = int(total_size * 0.7)
        test_size = int(total_size * 0.2)
        val_size = total_size - train_size - test_size

        logging.info(f"Train size: {train_size}, Test size: {test_size}, Validation size: {val_size}")

        train = data.take(train_size)
        val = data.skip(train_size).take(val_size)
        test = data.skip(train_size+val_size).take(test_size)

        save_dataset(train, self.data_transformation_config.train_data_path, "train")
        save_dataset(val, self.data_transformation_config.val_data_path, "val")
        save_dataset(test, self.data_transformation_config.test_data_path, "test")

        return train, val, test

       except Exception as e:
          raise CustomException(e, sys)
       
       
def save_dataset(dataset, directory, prefix):
    os.makedirs(directory, exist_ok=True)
    for idx, (image, label) in enumerate(dataset):
        for i in range(len(image)):
            image_path = os.path.join(directory, f"{prefix}_image_{idx * len(image) + i}.png")
            tf.keras.preprocessing.image.save_img(image_path, image[i])