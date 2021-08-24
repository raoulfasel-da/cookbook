# Google Sheet RFM pipeline

_Download link for necessary files_: [GSheet files](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/cookbook/tree/master/gsheet-rfm-pipeline/gsheet-rfm-pipeline)

In this recipe you will learn how to set up a pipeline that retrieves data from a Google Sheet, performs a simple RFM
analysis on it, and writes back the result to the Google Sheet.

## How does it work?

**Step 1:** Login to your UbiOps account at https://app.ubiops.com/ and create an API token with project editor
admin rights. To do so, click on *Users & permissions* in the navigation panel, and then click on *API tokens*.
Click on *create token* to create a new token.

![Creating an API token](../pictures/api_token_screenshot.png)

Give your new token a name, save the token in safe place and assign the following roles to the token: project editor and
blob admin. These roles can be assigned on project level.

**Step 2:** Download the [GSheet files](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/cookbook/tree/master/gsheet-rfm-pipeline/gsheet-rfm-pipeline) 
folder and open `gsheet.ipynb`. In the notebook you will find a space
to enter your API token and the name of your project in UbiOps. Paste the saved API token in the notebook in the
indicated spot and enter the name of the project in your UbiOps environment. This project name can be found in the top
of your screen in the WebApp.

**Step 3:** Run the Jupyter notebook *arthur* and everything will be 
automatically deployed to your UbiOps environment! 
Afterwards you can explore the code in the notebook or explore the application in the WebApp.
