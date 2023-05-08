
import cv2
import numpy as np
from PIL import Image
from keras_preprocessing import image
from flask import Flask, jsonify, request

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    filename = file.filename
    file.save(filename)
    img = cv2.imread(filename)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    lung_pixels = np.sum(markers == 2)
    total_pixels = np.prod(markers.shape)
    lung_percentage = lung_pixels / total_pixels * 100
    output=str(lung_percentage)
    print('Percentage of lungs affected: {:.2f}%'.format(lung_percentage))
    return output
if __name__ == '__main__':
    app.run(debug=True)
