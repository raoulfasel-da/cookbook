"""
The file containing the deployment code is required to be called 'deployment.py' and should contain the 'Deployment'
class and 'request' method.
"""

import os
import ubiops
import pickle


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

        # Setup a api config
        configuration = ubiops.Configuration()
        # Configure API key authorization using environment variables
        # https://ubiops.com/docs/deployments/environment-variables/
        configuration.api_key['Authorization'] = os.environ['YOUR_API_KEY']
        configuration.api_key_prefix['Authorization'] = ''

        # Defining host is optional and default to https://api.ubiops.com/v2.1
        configuration.host = "https://api.ubiops.com/v2.1"
        # Enter a context with an instance of the API client
        api_client = ubiops.ApiClient(configuration)

        # Create an instance of the API class
        self.api_instance = ubiops.CoreApi(api_client)

        print("Initialising blob storage Deployment")

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

        # Get the input number
        input_number = data.get('input_number')

        project_name = os.environ['PROJECT_NAME']
        blob_ttl = 86400  # one day

        # List latest blob
        api_response = self.api_instance.blobs_list(project_name, range=-1)

        # Default the old number to 0
        old_number = 0
        if len(api_response) > 0:
            # Get the last blob id
            blob_id = api_response[0].to_dict().get('id')

            # Get the latest blob using its id
            with self.api_instance.blobs_get(project_name, blob_id) as response:
                content = response.read()
                # We get the file as a bytestring so we can simply load it on the fly
                old_number = pickle.loads(content)

        # Add the number to the previous total
        output_number = old_number + input_number

        # Pickle the file to local storage
        # (Pickle is a serialisation method) https://docs.python.org/3/library/pickle.html
        pickle.dump(output_number, open("new_number.p", "wb"))

        # Upload the pickle
        self.api_instance.blobs_create(project_name, "new_number.p", blob_ttl=blob_ttl)

        # Print and return the current total
        print(output_number)

        return {
            "output_number": output_number
        }
