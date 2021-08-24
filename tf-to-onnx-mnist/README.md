# Tensorflow to ONNX

_Download link for necessary files_: [Tensorflow to ONNX files](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/cookbook/tree/master/tf-to-onnx-mnist/tf-to-onnx-mnist)

In this example we will show you the following:
How to convert a Tensorflow based image classification algorithm to ONNX and 
run it on UbiOps using the ONNX runtime.

## Overview of the Deployments

The resulting deployment is made up of the following:

| Deployment | Function |
|-------|----------|
| tf    | A deployment that uses a trained Tensorflow model |
| onnx  | The same model but now running on the ONNX runtime |

## How does it work?

**Step 1:** Login to your UbiOps account at https://app.ubiops.com/ and create an API token with project editor
admin rights. To do so, click on *Users & permissions* in the navigation panel, and then click on *API tokens*.
Click on *create token* to create a new token.

![Creating an API token](api_token_screenshot.png)

Give your new token a name, save the token in safe place and assign the following roles to the token: project editor and blob admin.
These roles can be assigned on project level.

_Download link for necessary files_: [Tensorflow to ONNX files](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/cookbook/tree/master/tf-to-onnx-mnist/tf-to-onnx-mnist) to enter your API token and the name of your project in UbiOps. Paste the saved API token in the notebook in the indicated spot
and enter the name of the project in your UbiOps environment. This project name can be found in the top of your screen in the
WebApp.

**Step 3:** Run the Jupyter notebook *tf-to-onnx-mnist* and everything will be 
automatically deployed to your UbiOps environment! 
Afterwards you can explore the code in the notebook or explore the application in the WebApp.
