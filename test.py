from __future__ import division
from libraries.nitramite_alpr.module import detect_model
from os.path import isfile, join
import matplotlib.pyplot as plt
from os import listdir
from PIL import Image
import numpy as np
import cv2
import os

# Paths
base_path = os.getcwd() + '/output/lp_training/'
model_output_path = base_path + 'model/'
base_test_path = base_path + 'test/'


def test(max_test_images):
    lp_model = detect_model.detect_model()
    lp_model.load_weights(model_output_path + 'lp_model.h5')

    files = [f for f in listdir(base_test_path) if isfile(join(base_test_path, f))]
    files = sorted(files)

    for i in range(max_test_images):
        img_name = files[i]  # test folder image index
        img = cv2.imread(base_test_path + img_name)
        inputs = Image.open(join(base_test_path, img_name))
        inputs = np.array(inputs, dtype=np.float32)
        inputs = inputs / 255.0
        inputs_net = np.array([inputs])
        outputs = lp_model.predict(inputs_net, batch_size=1, verbose=0, steps=None)
        outputs = outputs[0, :]

        x = int(outputs[0] * 256)
        y = int(outputs[1] * 256)

        w = int((outputs[2] - outputs[0]) * 256)
        h = int((outputs[3] - outputs[1]) * 256)

        print(str([x, y, w, h]))
        # plt.imshow(inputs)
        # plt.imshow(inputs[y:y + h, x:x + w, :])
        # plt.show()

        cv2.rectangle(img, (x, y), (x + w, y + h), 255, 1)
        cv2.imshow("lpTraining", img)
        cv2.waitKey(0)

