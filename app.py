import os
from flask import Flask, request, render_template
from src.Pipeline.predict_pipeline import PredictPipeline  

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


modelTrainer = PredictPipeline()

@app.route('/')
def home():
    return render_template('mushroom.html', prediction=None)

@app.route('/upload', methods=['POST'])
def upload_pic():
    if 'picInput' not in request.files:
        print("No file uploaded")
        return render_template('mushroom.html', error="No file uploaded")

    file = request.files['picInput']

    if file.filename == '':
        print("No file selected")
        return render_template('mushroom.html', error="No selected file")

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        print(f"File saved to: {file_path}")
        
        try:
            prediction = modelTrainer.predict(file_path)
            prediction['Edibility'] = "Edible" if prediction['Edibility'] == '1' else "Inedible"
            print(f"Prediction: {prediction}")
            return render_template('mushroom.html', prediction=prediction)
        except Exception as e:
            print(f"Prediction error: {e}")
            return render_template('mushroom.html', error=f"Prediction error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
