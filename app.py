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
    all_train_images = get_data(r'data\Train')
    all_test_images = get_data(r'data\Test')

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


if __name__ == '__main__':
    app.run(debug=True)
