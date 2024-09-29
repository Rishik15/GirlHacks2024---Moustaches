from flask import Flask, request, render_template, redirect, url_for
import pickle
from PIL import Image
import io
import os



# Open file read binary as object f
with open('mushroom_model.pkl', 'rb') as f:
    model = pickle.load(f)


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/upload', methods=['GET', 'POST'])
def upload_pic():
    
    ##User giving info
    if request.method == 'POST':
        
        #pic input is file id
        if 'picInput' not in request.files:
            return render_template('mushroom.html', error = "No file uploaded")
        
        file = request.files['picInput']

        #Weird case
        if file.filename == '':
            return render_template('No selected file', error = "No selected file")
        
        
        if file:
            image = Image.open(file)
            
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            
            file.save(file_path)
            
            #Rishik said no need to prep data
            
            
            prediction = model.predict(image)
            ##prediction = {'Ediblility':'1', 'Probability':45 }
            
            #convert from int to string that can be used in html
            prediction['Edibility'] = "Edible" if prediction['Edibility'] == 1 else "Inedible"
            
            
            ##Remove low chance shrooms
            print(prediction)
            
            #return render_template('mushroom.html', prediction)
            return render_template('mushroom.html', prediction=prediction)
            
           
        
    
        
    return render_template('mushroom.html')
    #GET 



if __name__== '__main__':
    app.run(debug=True)