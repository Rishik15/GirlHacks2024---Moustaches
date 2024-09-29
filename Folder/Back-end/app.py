from flask import Flask, request, render_template
from PIL import Image
import os
from src.Components.modelTraining import ModelTrainer  # Assuming ModelTrainer is defined in this module


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


model_trainer = ModelTrainer() 
model_trainer.load_model() 

@app.route('/upload', methods=['GET', 'POST'])
def upload_pic():
    if request.method == 'POST':
      
        if 'picInput' not in request.files:
            return render_template('mushroom.html', error="No file uploaded")
        
        file = request.files['picInput']


        if file.filename == '':
            return render_template('mushroom.html', error="No selected file")
        
        if file:
           
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            
        
            prediction = ModelTrainer.predict(file_path)

  
            prediction['Edibility'] = "Edible" if prediction['Edibility'] == '1' else "Inedible"
            

            return render_template('mushroom.html', prediction=prediction)
    

    return render_template('mushroom.html')

if __name__ == '__main__':
    app.run(debug=True)
