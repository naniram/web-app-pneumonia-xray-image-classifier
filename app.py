from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

app = Flask(__name__)

model = load_model('training/img_clf_model.h5')

def process_image(image_path, target_size=(256, 256)):
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return image, prediction

@app.route('/')
def upload_file():
    return render_template('upload.html') 

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file'] 
        img_path = 'static/uploaded_image.jpg'
        file.save(img_path) 
        image, prediction = process_image(img_path) 
        if prediction>=0.5:
            prediction_final = 'Pneumonia'
        else:
            prediction_final = 'Normal'
        return render_template('result.html', image_path=img_path, prediction_final= prediction_final, prediction=prediction[0][0])  # Render the result.html template and pass the prediction and image path as variables

if __name__ == '__main__':
    app.run(debug=True)
