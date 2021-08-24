"""
The file containing the deployment code is required to be called 'deployment.py' and should contain the 'Deployment'
class and 'request' method.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler



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
        #Load the dataset
        print("Loading data")
        diabetes_data = pd.read_csv(data["data"])
        
        # The data contains some zero values which make no sense (like 0 skin thickness or 0 BMI). 
        # The following columns/variables have invalid zero values:
        # glucosem bloodPressure, SkinThicknes, Insulin and BMI
        # We will replace these zeros with NaN and after that we will replace them with a suitable value.
        
        print("Imputing missing values")
        # Replacing with NaN
        diabetes_data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
        # Imputing NaN
        diabetes_data['Glucose'].fillna(diabetes_data['Glucose'].mean(), inplace = True)
        diabetes_data['BloodPressure'].fillna(diabetes_data['BloodPressure'].mean(), inplace = True)
        diabetes_data['SkinThickness'].fillna(diabetes_data['SkinThickness'].median(), inplace = True)
        diabetes_data['Insulin'].fillna(diabetes_data['Insulin'].median(), inplace = True)
        diabetes_data['BMI'].fillna(diabetes_data['BMI'].median(), inplace = True)
        
        # If this deployment is used for training, the target column
        # needs to be split from the data
        if data["training"] == True:
            X = diabetes_data.drop(["Outcome"], axis = 1) 
            y = diabetes_data.Outcome
        else:
            X = diabetes_data
            y = pd.DataFrame([1])
            
            
        print("Scaling data")
        # Since we are using a distance metric based algorithm we will use scikits standard scaler to scale all the features to [-1,1]
        sc_X = StandardScaler()
        X =  pd.DataFrame(sc_X.fit_transform(X,),
                columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
               'BMI', 'DiabetesPedigreeFunction', 'Age'])
        
        # UbiOps expects JSON serializable output or files, so we convert the dataframes to csv
        X.to_csv('X.csv', index = False)
        y.to_csv('y.csv', index = False, header = False)

        return {
            "cleaned_data": 'X.csv', "target_data": 'y.csv'
        }
