"""
The file containing the deployment code is required to be called 'deployment.py' and should contain the 'Deployment'
class and 'request' method.
"""

import os
import onnxruntime as rt
from imageio import imread
import numpy as np


class Deployment:

    def __init__(self, base_directory, context):
        self.sess = rt.InferenceSession("mnist.onnx")
        self.input_name = self.sess.get_inputs()[0].name

    def request(self, data):


        x = imread(data['image'])
        # convert to a 4D tensor to feed into our model
        x = x.reshape(1, 28, 28, 1)
        x = x.astype(np.float32) / 255

        print("Prediction being made")

        prediction = self.sess.run(None, {self.input_name: x})[0]

        return {'prediction': int(np.argmax(prediction)), 'probability': float(np.max(prediction))}

       
