from src.exception import CustomException
from src.logger import logging
import cv2
import imghdr
from dataclasses import dataclass
import os
import sys
import tensorflow as tf
import numpy as np
from src.Components.datatransformation import DataTransformation
from src.Components.modelTraining import ModelTrainer


class dataIngestion:
    def __init__(self) -> None:
       pass
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            logging.info("Handling dodgy images.")
            data_dir = 'data/raw'

            handleDodgyImage(data_dir = data_dir)

            class_names = ['non_edible', 'edible']

            data = tf.keras.utils.image_dataset_from_directory(
                data_dir,
                image_size=(256, 256),  
                batch_size=32,         
                label_mode='int',      
                class_names=class_names, 
                shuffle=True           
            )

            data_iterator = data.as_numpy_iterator()

            batch = data_iterator.next()
            logging.info(f"Batch shape: {batch[0].shape}")

            return data


        except Exception as e:
            raise CustomException(e, sys)
        

def handleDodgyImage(data_dir: str):
    image_exts = ['jpeg', 'jpg', 'bmp', 'png']
    for image_class in os.listdir(data_dir):
        class_path = os.path.join(data_dir, image_class)
        if not os.path.isdir(class_path):
            continue
        for image in os.listdir(os.path.join(data_dir, image_class)):
            image_path = os.path.join(data_dir, image_class, image)
            if os.path.isdir(image_path):
                continue
            try:
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_exts:
                    print(f'Image not in extensions list {image_path}')
                    os.remove(image_path)
            except Exception as e:
                raise CustomException(e, sys)
            

if __name__ == "__main__":
    try:
        logging.info("Starting the full pipeline...")

        # Step 1: Data Ingestion
        ingestion = dataIngestion()
        data = ingestion.initiate_data_ingestion()

        # Step 2: Data Transformation
        transformation = DataTransformation()
        train_data, val_data, test_data = transformation.initiate_data_transformation(data)

        # Step 3: Model Training
        trainer = ModelTrainer()
        trainer.initiate_model_trainer(train_data, val_data, test_data)

        logging.info("Pipeline executed successfully.")

    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")
        raise CustomException(e, sys)