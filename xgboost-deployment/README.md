# XGboost model template

_Download link for necessary files_: [XGBoost files](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/cookbook/tree/master/xgboost-deployment/xgboost-recipe)

In this example we will show you the following:

How to create a deployment that uses a built XGboost model to make predictions on the price of houses based on criteria from [the House Sales in King County, USA Dataset](https://www.kaggle.com/harlfoxem/housesalesprediction).

## XGboost model

The resulting deployment is made up of the following:

| Deployment | Function |
|-------|----------|
| xgboost-deployment | A deployment that uses a trained XGboost model to predict house prices based on house criteria |


## How does it work?

**Step 1:** Login to your UbiOps account at https://app.ubiops.com/ and create an API token with project editor
 rights. To do so, click on *Users & permissions* in the navigation panel, and then click on *API tokens*.
Click on *create token* to create a new token.

![Creating an API token](api_token_screenshot.png)

Give your new token a name, save the token in safe place and assign the following role to the token: project editor.
This role can be assigned on project level.

**Step 2:** Download the *[xgboost-recipe](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/cookbook/tree/master/xgboost-deployment/xgboost-recipe)* folder and open `xgboost_template.ipynb`. In the notebook you will find a space to enter your API token and the name of your project in UbiOps. Paste the saved API token in the notebook in the indicated spot and enter the name of the project in your UbiOps environment. This project name can be found in the top of your screen in the WebApp. In the image in step 1 the project name is *scikit-example*.

**Step 3:** Run the Jupyter notebook `xgboost_template.ipynb` and everything will be automatically deployed to your UbiOps environment! 
Afterwards you can explore the code in the notebook or explore the application in the WebApp.
