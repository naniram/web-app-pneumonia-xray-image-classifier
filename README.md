# Pneumonia Xray image classifier

It is a web-based application that uses a deep learning model to classify chest X-ray images as either pneumonia or normal. The application is built with Flask, and the model is trained using TensorFlow.The dataset can be downloaded from kaggle: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## Model Description

The classifier is based on a convolutional neural network (pre-trained VGG16) trained on a dataset of more than 5000 chest X-ray images . The model has been saved in the `img_clf_model.h5` file and is loaded by the Flask application to make predictions on uploaded images.

## Features

- Upload chest X-ray images through a web interface.
- Predict if the uploaded X-ray shows signs of pneumonia or is normal.
- Display the uploaded image along with the prediction result.


## How to Run the App
Run the Flask application: 'python app.py'  
Open your web browser and go to http://127.0.0.1:5000    
On the homepage, upload a chest X-ray image using the provided form.  
Click the "Upload" button.  
The application will display the uploaded image and the prediction result (either "Pneumonia" or "Normal").

