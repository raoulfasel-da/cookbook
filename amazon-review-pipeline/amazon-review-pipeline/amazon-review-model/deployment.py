"""
The file containing the deployment code is required to be called 'deployment.py' and should contain the 'Deployment'
class and 'request' method.
"""

import logging
import os
import pandas as pd
import joblib
import ubiops


logger = logging.getLogger('Amazon review model')


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

        logger.info("Initialising model")
        self.model = joblib.load('amazon_review_model.pkl')
        self.count_vectorizer = joblib.load('count_vectorizer.pkl')

    @staticmethod
    def _connect_api(api_token):
        client = ubiops.ApiClient(
            ubiops.Configuration(api_key={'Authorization': api_token}, host='https://api.ubiops.com/v2.1')
        )
        return ubiops.CoreApi(client)

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

        # Load the data of the day into a dataframe
        df = pd.read_json(data)

        # From the dataframe we can get the day
        day = df['day_of_week'].iloc[0]

        # Let's get those predictions and take the averages of all review scores
        average_predictions = self.return_predictions(dataframe=df)

        # Get yesterday's day
        days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        index = days.index(day)
        yester_day_index = index - 1
        yesterday = days[yester_day_index]

        # Then retrieve yesterday's average review scores
        # If those are not here (because it was a weekend day),
        # we will compare the predicted review score to a set threshold
        review_scores = self.retrieve_review_scores(yesterday=yesterday)

        below_threshold = 0
        products_below_threshold = []

        # Now we are ready to see if the review scores are stable
        # For every product in the predictions, compare the average review score to the threshold score
        for index, value in average_predictions.items():

            if review_scores is not None:
                review_threshold = review_scores.loc[review_scores['product_id'] == index, 'predictions'].iloc[0]
            else:
                # Load in the minimal threshold for the review
                review_threshold = float(os.environ['THRESHOLD'])
                
            # Prompt a signal if the value drops significantly (0.2) below the threshold
            if value < (review_threshold - 0.2):
                below_threshold += 1
                product_title = df.loc[df['product_id'] == index, 'product_name'].iloc[0]
                products_below_threshold.append(product_title, )
                logger.error(f"Product with id {index}, has not met its review standards")

        # Output the review_scores to a csv, so it will be available for the next request.
        average_predictions.to_csv(f"review_scores_{day.lower()}.csv", index=True)

        # Return the results
        return {
            'total_products': len(average_predictions),
            'below_threshold': below_threshold,
            'products_below_threshold': products_below_threshold,
            'review_scores': f"review_scores_{day.lower()}.csv"
        }

    def return_predictions(self, dataframe):

        # The test column of the data is the 'review' column
        x_test = dataframe['review']

        # Because our X test data is text, it has to be transformed to vectors to be able to make predictions on
        x_test_transformed = self.count_vectorizer.transform(x_test)

        # Let's predict and paste the resulting array right onto the existing dataframe
        dataframe['predictions'] = self.model.predict(x_test_transformed)

        # Generate a new series with the average of the predictions per product
        return dataframe.groupby('product_id')['predictions'].mean()

    def retrieve_review_scores(self, yesterday):

        # Connect to APi to retrieve the review scores file from UbiOps
        api = self._connect_api(api_token=os.environ['API_TOKEN'])

        # If the review score of yesterday is there retrieve it from UbiOps
        project_name = os.environ['PROJECT_NAME']
        blobs_list = api.blobs_list(project_name=project_name)
        review_scores = None

        for blob in blobs_list:
            if blob.filename == f"review_scores_{yesterday.lower()}.csv":
                response = api.blobs_get(project_name=project_name, blob_id=str(blob.id))
                review_scores = pd.read_csv(response)

        return review_scores
