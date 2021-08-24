"""
The file containing the deployment code is required to be called 'deployment.py' and should contain the 'Deployment'
class and 'request' method.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from joblib import dump
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

class Deployment:

    def __init__(self, base_directory, context):
        """
        Initialisation method for the deployment. It can for example be used for loading modules that have to be kept in
        memory or setting up connections. Load your external model files (such as pickles or .h5 files) here.

        :param str base_directory: absolute path to the directory where the deployment.py file is located
        :param dict context: a dictionary containing details of the deployment that might be useful in your code.
            It contains the following keys:
                - deployment (str): name of the deployment
                - version (str): name of the version
                - input_type (str): deployment input type, either 'structured' or 'plain'
                - output_type (str): deployment output type, either 'structured' or 'plain'
                - language (str): programming language the deployment is running
                - environment_variables (str): the custom environment variables configured for the deployment.
                    You can also access those as normal environment variables via os.environ
        """

        print("Initialising My Deployment")

    def request(self, data):
        """
        Method for deployment requests, called separately for each individual request.

        :param dict/str data: request input data. In case of deployments with structured data, a Python dictionary
            with as keys the input fields as defined upon deployment creation via the platform. In case of a deployment
            with plain input, it is a string.
        :return dict/str: request output. In case of deployments with structured output data, a Python dictionary
            with as keys the output fields as defined upon deployment creation via the platform. In case of a deployment
            with plain output, it is a string. In this example, a dictionary with the key: output.
        """

        print("Processing request for My Deployment")
        # Load the dataset
        print("Loading data")
        
        X = pd.read_csv(data["cleaned_data"])
        y = pd.read_csv(data["target_data"], header = None)
        print(X.shape)
        print(y.shape)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42, stratify=y)

        
        # Setup a knn classifier with k neighbors
        knn = KNeighborsClassifier(n_neighbors=7) 
        
        # Fit the model on training data
        knn.fit(X_train,y_train)
        
        # Get accuracy on test set. Note: In case of classification algorithms score method represents accuracy.
        score = knn.score(X_test,y_test)
        print('KNN accuracy: ' + str(score))
        
        # let us get the predictions using the classifier we had fit above
        y_pred = knn.predict(X_test)
                
        # Output classification report
        print('Classification report:')
        print(classification_report(y_test,y_pred))
        
        # Persisting the model for use in UbiOps
        with open('knn.joblib', 'wb') as f:
           dump(knn, 'knn.joblib')
        
        
        return {
            "trained_model": 'knn.joblib', "model_score": score
        }
