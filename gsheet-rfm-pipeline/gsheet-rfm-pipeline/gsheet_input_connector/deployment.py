"""
The file containing the deployment code is required to be called 'deployment.py' and should contain the 'Deployment'
class and 'request' method.
"""

import os
import json
from google.oauth2 import service_account
import pygsheets
from joblib import dump


class Deployment:

    def __init__(self, base_directory, context):

        print('Initialising the connection to the google drive')
        self.gc = None

        SCOPES = ('https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive')
        service_account_info = json.loads(os.environ['credentials'])
        my_credentials = service_account.Credentials.from_service_account_info(service_account_info, scopes=SCOPES)

        try:
            self.gc = pygsheets.authorize(custom_credentials=my_credentials)
            print('Established succesfull connection')
        except Exception as e:
            print('Connection failed, ', e.__class__, 'occurred.')

    def request(self, data):

        print('Getting the requested file')
        spreadsheet = self.gc.open(data['filename'])
        sheet_data = spreadsheet[0]

        # UbiOps expects JSON serializable output or files, so we pickle the data
        with open('tmp_sheet.joblib', 'wb') as f:
           dump(sheet_data, 'tmp_sheet.joblib')
        
        return {'data': 'tmp_sheet.joblib'}