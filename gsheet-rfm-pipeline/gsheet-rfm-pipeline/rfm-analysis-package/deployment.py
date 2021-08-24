"""
The file containing the deployment code is required to be called 'deployment.py' and should contain the 'Deployment'
class and 'request' method.
"""

import pygsheets
from joblib import load, dump
import pandas as pd


class Deployment:

    def __init__(self, base_directory, context):

        print('Initalizing model')

    def request(self, data):

        print('Loading the data')
        sheet_data = load(data['retail_data'])

        # Transforming it into a Pandas DataFrame
        data_df = sheet_data.get_as_df()

        # RFM analyis
        print('Performing RFM analysis')
        data_df['TotalPrice'] = data_df['Quantity'].astype(int) * data_df['UnitPrice'].astype(float)
        data_df['InvoiceDate'] = pd.to_datetime(data_df['InvoiceDate'])

        rfm= data_df.groupby('CustomerID').agg({'InvoiceDate': lambda date: (date.max() - date.min()).days,
                                                'InvoiceNo': lambda num: len(num),
                                                'TotalPrice': lambda price: price.sum()})

        # Change the name of columns
        rfm.columns=['recency','frequency','monetary']

        # Computing Quantile of RFM values
        rfm['recency'] = rfm['recency'].astype(int)
        rfm['r_quartile'] = pd.qcut(rfm['recency'].rank(method='first'), 4, ['1','2','3','4']).astype(int)
        rfm['f_quartile'] = pd.qcut(rfm['frequency'], 4, ['4','3','2','1']).astype(int)
        rfm['m_quartile'] = pd.qcut(rfm['monetary'], 4, ['4','3','2','1']).astype(int)

        rfm['RFM_Score'] = rfm.r_quartile.astype(str)+ rfm.f_quartile.astype(str) + rfm.m_quartile.astype(str)

        # Filter out Top/Best customers
        print('Filtering out top customers')
        top_customers = rfm[rfm['RFM_Score']=='111'].sort_values('monetary', ascending=False)        

        # UbiOps expects JSON serializable output or files, so we pickle the data
        with open('top_customers.joblib', 'wb') as f:
           dump(top_customers, 'top_customers.joblib')
        
        return {'top_customers': 'top_customers.joblib'}