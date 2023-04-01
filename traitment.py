import os
from typing import Any

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.models import load_model

import cv2

class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
               'W', 'X', 'Y']

prediction_class_converter = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "K",
    10: "L",
    11: "M",
    12: "N",
    13: "O",
    14: "P",
    15: "Q",
    16: "R",
    17: "S",
    18: "T",
    19: "U",
    20: "V",
    21: "W",
    22: "X",
    23: "Y"
}


def get_data(images_dir: str) -> list[tuple[Any, int]]:
    data = []

    for class_label, label in enumerate(class_names):
        path = os.path.join(images_dir, label)
        print(path)
        for img in os.listdir(path):
            full_path_image = os.path.join(path, img)
            image = cv2.imread(full_path_image)
            data.append((image, class_label))

    return data


def append_data(images_dir):
    x_list = []
    y_list = []
    for img, label in images_dir:
        x_list.append(img)
        y_list.append(label)
    return x_list, y_list


def create_trained_model(x_train, xtest, y_train, ytest) -> bool:
    try:
        # create model
        model = Sequential()
        model.add(Conv2D(24, (3, 3), activation="relu", input_shape=(28, 28, 3)))
        model.add(MaxPool2D())

        model.add(Conv2D(24, (3, 3), activation="relu"))
        model.add(MaxPool2D())

        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPool2D())
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(24, activation="softmax"))

        model.compile(optimizer='adam',
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

        # use early stopping to avoid over fitting
        early_stop = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20)

        # train model
        model.fit(x=x_train, y=y_train, epochs=400, validation_data=(xtest, ytest), callbacks=[early_stop])
    except ValueError:
        return False
    else:
        model.save('sign_language_interpreter_model.h5')
        return True


def test_loss(xtest, ytest):
    sign_model = load_model('sign_language_interpreter_model.h5')
    return sign_model.evaluate(xtest, ytest)


def prediction_image(gray_image):
    # load saved model
    sign_model = load_model('sign_language_interpreter_model.h5')
    # transform the grayscale image to 3 canal
    gray_image_3_canal = cv2.merge((gray_image, gray_image, gray_image))
    # resize the original gray image
    gray_image_resized = cv2.resize(gray_image_3_canal, (28, 28), interpolation=cv2.INTER_LINEAR)
    # normalize the image
    gray_image_normalized = gray_image_resized / 255
    gray_image_normalized = (np.expand_dims(gray_image_normalized, 0))

    prediction_value_array = sign_model.predict(gray_image_normalized)

    number_class_predict = np.argmax(prediction_value_array)

    return prediction_class_converter[number_class_predict]