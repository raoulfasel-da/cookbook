"""
The file containing the deployment code is required to be called 'deployment.py' and should contain the 'Deployment'
class and 'request' method.
"""

import os
import json
from google.oauth2 import service_account
import pygsheets
from joblib import load
import pandas


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

        print('Loading top customers')
        top_customers = load(data['data'])

        print('Inserting data into the google sheet')
        spreadsheet = self.gc.open(os.environ['filename'])
        sheet_title = os.environ['sheet_title']

        try:
            sh = spreadsheet.worksheet_by_title(sheet_title)
        except:
            print('Worksheet does not exist, adding new sheet')
            spreadsheet.add_worksheet(sheet_title)
            sh = spreadsheet.worksheet_by_title(sheet_title)
        finally:
            sh.set_dataframe(top_customers, 'A1', copy_index = True)
            sh.update_value('A1', 'CustomerID')
            print('Data inserted successfully')     
        