# MLFlow UbiOps

_Download link for necessary files_: [MLFlow files](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/cookbook/tree/master/mlflow-example/mlflow-recipe)

In this example we will show you the following:
How to train a model the predicts the quality of wine based on some parameters, then test for the optimal parameters using the MLFlow tool and then deploy it to the UbiOps environment.


## MLFlow Deployment

The resulting deployment is made up of the following:

| Deployment | Function |
|-------|----------|
| mlflow-deployment | A deployment that uses a trained AI model to predict the quality of wine |



## How does it work?

**Step 1:** Login to your UbiOps account at https://app.ubiops.com/ and create an API token with project editor
 rights. To do so, click on *Users & permissions* in the navigation panel, and then click on *API tokens*.
Click on *create token* to create a new token.

![Creating an API token](../pictures/api_token_screenshot.png)

Give your new token a name, save the token in safe place and assign the following role to the token: project editor.
This role can be assigned on project level.

**Step 2:** Download the [mlflow-example](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/cookbook/tree/master/mlflow-example/mlflow-recipe) folder and open `mlflow_example.ipynb`. In the notebook you will find a space
to enter your API token and the name of your project in UbiOps. Paste the saved API token in the notebook in the indicated spot
and enter the name of the project in your UbiOps environment. This project name can be found in the top of your screen in the
WebApp.

**Step 3:** Run the Jupyter notebook *mlflow_example* and everything will be automatically deployed to your UbiOps environment! 
Afterwards you can explore the code in the notebook or explore the application in the WebApp.
