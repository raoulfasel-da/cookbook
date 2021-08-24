from setup_logging import setup_logging
import logging
import sys
sys.path.append('/usr/lib/python3/dist-packages') # We need to point to the location where Caffe installs its Python lib
import os
import cv2
import caffe
import numpy as np
from PIL import Image
import base64
import io


def base64_to_image(enc_str):
    """
    Decodes a base64 string to an image and returns it as a Numpy array.
    The image will be resized using OpenCV to a resolution of 224x224 pixels.
    """
    dec_str = base64.b64decode(str(enc_str))
    img = Image.open(io.BytesIO(dec_str))
    img_arr = np.asarray(img)
    res = cv2.resize(img_arr, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

    return res


class Deployment:

    def __init__(self, base_directory):
        """
        Initialisation method for the deployment. This will be called at start-up of the model in UbiOps.

        :param str base_directory: absolute path to the directory where this file is located.
        """

        setup_logging()
        logging.info("Initialising Caffe model")

        # Here we initialize the Caffe classifier model with the pre-trained files
        model_def_file = os.path.join(base_directory, "age.prototxt")
        caffe_model = os.path.join(base_directory, "dex_chalearn_iccv2015.caffemodel")

        self.net = caffe.Classifier(model_def_file, caffe_model)


    def request(self, data):
        """
        Method for model requests, called for every individual request

        :param dict data: dictionary with the model data. In this case it will hold a key 'photo' with a base64 string
        as value.
        :return dict prediction: JSON serializable dictionary with the output fields as defined on model creation
        """

        logging.info("Processing model request")

        # Convert the base64 string input to a Numpy array of the right format. Using the function defined above.
        photo_data = base64_to_image(data['photo'])

        # Call the predict function of the Caffe net
        out = self.net.predict([photo_data], oversample=False)

        # From the output array we return the index of the largest value. The out array holds probabilities for
        # ages 1-100.
        age = out[0].argmax()

        # Here we return a JSON with the estimated age as integer
        return {'age': int(age)}
