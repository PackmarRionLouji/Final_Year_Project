from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
import io
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    filename = file.filename
    file.save(filename)
    model = load_model('pneumonia_detection_model (1).h5')
    test_image = image.load_img(filename, target_size = (256,256))
    #img = cv2.imread(test_image)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = cv2.resize(img, (256, 256))
    #img = np.expand_dims(img, axis=0)
    #img = img / 255.0
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    prediction = model.predict(test_image)[0][0]
    threshold = 0.5 # Set threshold for predicting pneumonia
    if prediction > threshold:
        print('Pneumonia detected.')
    else:
        print('Pneumonia not detected.')
    
    output = prediction.tolist()
    return str(output)
    
if __name__ == '__main__':
    app.run(debug=True)

