from flask import Flask, jsonify, request
from typing import Any
from traitment import get_data, append_data, create_trained_model, test_loss, prediction_image

import numpy as np
import glob
import json
import os
import cv2

app = Flask(__name__)

all_train_images = list[tuple[Any, int]]
all_test_images = list[tuple[Any, int]]

X_train = list()
X_test = list()
y_train = list()
y_test = list()

if not len(glob.glob('*.h5')):
    all_train_images = get_data(r'data/Train')
    all_test_images = get_data(r'data/Test')

    X_train, y_train = append_data(all_train_images)
    X_test, y_test = append_data(all_test_images)

    # Normalization
    X_train = np.array(X_train) / 255
    X_test = np.array(X_test) / 255

    y_train = np.array(y_train)
    y_test = np.array(y_test)


@app.route('/')
def alive():
    return jsonify({'response': 'server sign language is alive'})


@app.route('/train', methods=['GET'])
def train_model():
    if not len(glob.glob('*.h5')):
        if create_trained_model(X_train, X_test, y_train, y_test):
            return jsonify({'model': 'created successfully'}), 200
        return jsonify({'model': 'not created'}), 500
    return jsonify({'model': 'already created'}), 200


@app.route('/prediction', methods=['POST'])
def prediction():
    if request.method == 'POST':
        try:
            frame = request.data
            image = np.asarray(bytearray(frame), dtype="uint8")
            # get gray image
            image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

            class_predict = prediction_image(image)

            return jsonify({'response': class_predict}), 200
        except cv2.error:
            return jsonify({'response': '#'}), 404


@app.route('/accuracy', methods=['GET'])
def accuracy():
    if not os.path.exists('accuracy_json.txt'):
        val = test_loss(X_test, y_test)
        with open('accuracy_json.txt', 'x') as f:
            json.dump({'accuracy': val[1]}, f, indent=4)

    with open('accuracy_json.txt', 'r') as f:
        accuracy_dict = json.load(f)

    return jsonify({'accuracy': accuracy_dict['accuracy']}), 200


@app.route('/name', methods=['GET'])
def model_name():
    name = glob.glob('*.h5')
    return jsonify({'name': name[0]})


if __name__ == '__main__':
    app.run(debug=True)
