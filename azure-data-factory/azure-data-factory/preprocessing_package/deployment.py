import json
import pandas as pd

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

        print("Initialising preprocessing Deployment")

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

        print("Processing request for preprocessing Deployment")

        print("Loading data")
        rows = json.loads(data["data"])

        df = pd.json_normalize(rows)
        df.to_csv('input.csv', index=False, encoding='utf-8')
        diabetes_data = pd.read_csv('input.csv')

        print(open('input.csv', 'r').read())

        # If this deployment is used for training, the target column
        # needs to be split from the data
        if data["training"] == True:
            X = diabetes_data.drop(["Outcome"], axis=1)
            y = diabetes_data.Outcome
        else:
            X = diabetes_data
            y = pd.DataFrame([1])

        print("Scaling data")
        # Since we are using a distance metric based algorithm we will use sci-kits standard scaler
        # to scale all the features to [-1,1]
        columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction',
            'Age'
        ]
        sc_X = StandardScaler()
        X = pd.DataFrame(sc_X.fit_transform(X, ), columns=columns)

        # UbiOps expects JSON serializable output or files, so we convert the dataframe to csv
        X.to_csv('X.csv', index=False)
        y.to_csv('y.csv', index=False, header=False)

        return {
            "cleaned_data": 'X.csv', "target_data": 'y.csv'
        }

