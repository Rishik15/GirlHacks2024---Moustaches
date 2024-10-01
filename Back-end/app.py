import os
from flask import Flask, request, render_template
from PIL import Image
from src.Components.modelTraining import ModelTrainer  
from tensorflow.keras.models import load_model
import tensorflow as tf

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model = None
model_trainer = ModelTrainer() 

@app.before_first_request
def load_model_on_start():
    global model
    try:
        model_trainer.load_model()  
        model = model_trainer.model
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    except Exception as e:
        print(f"Error loading model: {e}")


@app.route('/upload', methods=['GET', 'POST'])
def upload_pic():
    if request.method == 'POST':
      
        if 'picInput' not in request.files:
            return render_template('mushroom.html', error="No file uploaded")
        
        file = request.files['picInput']

        if file.filename == '':
            return render_template('mushroom.html', error="No selected file")
        
        if file:
            try:
                img = Image.open(file)
            except Exception as e:
                return render_template('mushroom.html', error="Invalid image file")

            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            
            try:
                prediction = model_trainer.predict(file_path) 
            except Exception as e:
                return render_template('mushroom.html', error=f"Prediction error: {e}")
  
            prediction['Edibility'] = "Edible" if prediction['Edibility'] == '1' else "Inedible"
            
            return render_template('mushroom.html', prediction=prediction)

    return render_template('mushroom.html')


if __name__ == '__main__':
    app.run(debug=True)
