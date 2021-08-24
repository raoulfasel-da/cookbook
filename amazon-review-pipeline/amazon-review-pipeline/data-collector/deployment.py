"""
The file containing the deployment code is required to be called 'deployment.py' and should contain the 'Deployment'
class and 'request' method.
"""

import logging
import os
from datetime import datetime
import pandas as pd

logger = logging.getLogger('Data collector')


def return_days(n):
    return datetime.strptime(n, '%Y-%m-%d %H:%M:%S').strftime('%A')


class InvalidDateError(Exception):
    pass


class Deployment:

    def __init__(self, base_directory):
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

        logger.info("Initialising Data collector")

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

        # Get today's day if defined in environment variables, otherwise take today's day
        day = os.environ.get('DAY', datetime.today().strftime('%A')).title()

        # Raise an error if the day is invalid (there's no data for saturday/sunday)
        if day not in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            raise InvalidDateError(f"No reviews found for day {day}")

        # Read in the reviews as a dataframe
        reviews = pd.read_csv('reviews.csv')

        # Reformat the time of review into day_of_week so we can filter
        reviews['day_of_week'] = reviews['time_of_review'].apply(return_days)

        # Only collect data from `day`
        logger.info(f"Collecting data from {day}")
        day_reviews = reviews[reviews['day_of_week'] == day]

        logger.info(f"Data retrieved successfully, sending data to model")
        df_string = day_reviews.to_json()

        # Output the resulting dataframe (in JSON) to the model
        return df_string
