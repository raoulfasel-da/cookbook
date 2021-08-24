# Twitter sentiment analysis UbiOps

_Download link for necessary files_: [Twitter sentiment analysis](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/cookbook/tree/master/twitter-sentiment-analysis/twitter-sentiment-analysis)

In this example we will show you the following:
- How to connect with the Twitter API to collect tweets with a certain hashtag and predict the sentiment of. 
- How to display the sentiment of these tweets in a google sheets, that for example Tableau can use to read from. 



# Twitter sentiment analysis Deployment

The resulting deployment is made up of the following:

| Deployment | Function |
|------------|----------|
| sentiment-deployment | A deployment that uses a trained sentiment analysis model to predict sentiment of tweets around a user inputted hash tag and inserts the results in a Google sheet |


## How does it work?

**Step 1:** Login to your UbiOps account at https://app.ubiops.com/ and create an API token with project editor rights. 
To do so, click on *Users & permissions* in the navigation panel, and then click on *API tokens*.
Click on *create token* to create a new token.

![Creating an API token](api_token_screenshot.png)

Give your new token a name, save the token in safe place and assign the following role to the token: project editor.
This role can be assigned on project level.

**step 2:** Follow the next steps to get the necessary Twitter tokens and Google credentials: 
1. Create a [service user](https://cloud.google.com/iam/docs/creating-managing-service-accounts#creating) in Google.
2. Create [credentials](https://cloud.google.com/iam/docs/creating-managing-service-account-keys#creating_service_account_keys) for the service user (called “keys” here).
3. [Share](https://robocorp.com/docs/development-guide/google-sheets/interacting-with-google-sheets#create-a-new-google-sheet-and-add-the-service-account-as-an-editor-to-it) the Google sheet with the Google service user account just like you would with a normal user: You hereby give it permission to edit your sheet.
4. A Twitter [developer account](https://developer.twitter.com/en/apply-for-access) and access to the Twitter API.

**Step 3:** Download the [Twitter sentiment analysis](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/cookbook/tree/master/twitter-sentiment-analysis/twitter-sentiment-analysis) folder and open `twitter_sentiment_analysis.ipynb`. In the notebook you will find a space
to enter your API token and the name of your project in UbiOps. Paste the saved API token in the notebook in the indicated spot
and enter the name of the project in your UbiOps environment. This project name can be found in the top of your screen in the
WebApp.

**Step 4:** Run the Jupyter notebook *twitter_sentiment_analysis* and everything will be automatically deployed to your UbiOps environment! 
Afterwards you can explore the code in the notebook or explore the application in the WebApp.
