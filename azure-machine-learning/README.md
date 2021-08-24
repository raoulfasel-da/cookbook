# Azure Machine Learning - UbiOps integration
**Note**: This notebook runs on Python 3.6.

_Download link for necessary files_: [Azure ML services example](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/cookbook/tree/master/azure-machine-learning/azure-machine-learning).

In this notebook we will show you:

- how to train a model on Azure ML

- how to deploy that model on UbiOps

For this example we will train a model on the MNIST dataset with Azure ML services and then deploy the trained model on UbiOps. Parts of this notebook were directly taken from [Azure documentation](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-train-models-with-aml), which can be found as another notebook [here](https://github.com/Azure/MachineLearningNotebooks/blob/master/tutorials/image-classification-mnist-data/img-classification-part1-training.ipynb). 
The trained model can be adapted to your usecase. The MNIST model is taken merely to illustrate how a model trained with Azure ML services could be converted to run on UbiOps. 


NOTE: Make sure the jupyter notebook is running from an environment with the requirements (see requirements.txt) installed.
Also provide the Azure config.json in the `config` folder.

## How does it work?

**Step 1:** Login to your UbiOps account at https://app.ubiops.com/ and create an API token with project editor 
rights. To do so, click on *Users & permissions* in the navigation panel, and then click on *API tokens*.
Click on *create token* to create a new token.

![Creating an API token](api_token_screenshot.png)

Give your new token a name, save the token in safe place and assign the 'project-editor' role to the token.
The role can be assigned on project level.

**Step 2:** Download the [azure-machine-learning](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/cookbook/tree/master/azure-machine-learning/azure-machine-learning) folder and open *azure_ml.ipynb*. In the notebook you will find a space
to enter your API token and the name of your project in UbiOps. Paste the saved API token in the notebook in the indicated spot
and enter the name of the project in your UbiOps environment. This project name can be found ar the top of your screen in the
WebApp. In the image in step 1 the project name is *scikit-example*.

**Step 3:** Run the Jupyter notebook *azure_ml* and everything will be automatically deployed to your Azure and UbiOps environment! 
Afterwards you can explore the code in the notebook or explore the application in the WebApp.
