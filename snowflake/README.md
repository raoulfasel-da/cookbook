# Snowflake UbiOps

_Download link for necessary files_: [Snowflake files](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/cookbook/tree/master/snowflake/snowflake-ubiops-example)

In this example we will show you the following:
How to create a deployment that makes connection with the snowflake environment and retrieves data on request. The user will be able to send a request with the amount of money he has and the deployment will check our (fake) shop to see what he can afford.


## Snowflake Deployment

The resulting deployment is made up of the following:

| Deployment | Function |
|-------|----------|
| snowflake-deployment | A deployment that uses input from the user (price) to check in the snowflake database the items he can afford  |


## How does it work?

**Step 1:** Login to your UbiOps account at https://app.ubiops.com/ and create an API token with project editor
 rights. To do so, click on *Users & permissions* in the navigation panel, and then click on *API tokens*.
Click on *create token* to create a new token.

![Creating an API token](api_token_screenshot.png)

Give your new token a name, save the token in safe place and assign the following role to the token: project editor.
This role can be assigned on project level.

**Step 2:** Download the [snowflake-ubiops-example](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/cookbook/tree/master/snowflake/snowflake-ubiops-example) folder and open `snowflake.ipynb`. In the notebook you will find a space
to enter your API token and the name of your project in UbiOps. Paste the saved API token in the notebook in the indicated spot
and enter the name of the project in your UbiOps environment. This project name can be found in the top of your screen in the
WebApp.

**Step 3:** Run the Jupyter notebook *snowflake.ipynb* and everything will be automatically deployed to your UbiOps environment! 
Afterwards you can explore the code in the notebook or explore the application in the WebApp.
