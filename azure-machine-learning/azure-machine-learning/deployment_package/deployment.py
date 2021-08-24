import os
from joblib import load
from imageio import imread
import numpy as np


class Deployment:

    def __init__(self, base_directory):
        """
        Initialisation method for the deployment. It can for example be used for loading modules that have to be kept in
        memory or setting up connections. Load your external model files (such as pickles or .h5 files) here.

        :param str base_directory: absolute path to the directory where the deployment.py file is located.
        """

        # Initialize the trained model
        print("Initializing MNIST model")
        mnist_model = os.path.join(base_directory, "sklearn_mnist_model.pkl")
        self.model = load(mnist_model)

    def request(self, data):
        """
        Method for model requests, called separately for each individual request.

        :param dict data: Python dictionary with deployment input data. In case of deployments with structured data,
        dict keys are the input fields of the deployment as defined upon deployment creation via the platform.

        :return dict prediction: Python dictionary with the output fields as defined on deployment creation.
        """

        print("Processing request for MNIST model")

        # Read in data
        x = imread(data['image'])
        # Convert to a 2D tensor to feed into the model
        x = x.reshape(1, 28 * 28)
        x = x.astype(np.uint8) / 255
        result = self.model.predict(x)

        # Output prediction result
        return {'prediction': int(result)}
