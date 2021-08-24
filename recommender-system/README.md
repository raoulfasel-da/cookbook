# Making a recommender model and deploying it to UbiOps

_Download link for necessary files_: [Recommender model files](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/cookbook/tree/master/recommender-system/recommender-system)

!!! info "This recipe is related to a blogpost"
    In a blogpost we explain how to put this recommender model behind a WebApp. You can read it
    [here](https://ubiops.com/how-to-build-and-implement-a-recommendation-system-from-scratch-in-python/).

In this example we will show you the following:

- how to train a recommender model on shopping data using the Apriori algorithm

- How to deploy that model to UbiOps

Recommender models are everywhere nowadays. At every webshop you will receive suggestions based on products you have viewed or added to your shopping cart. In this cookbook recipe we will make such a recommender model that can be used in the backend of a webshop.

## The model itself

The recommender model is created using [the Apriori algorithm](https://en.wikipedia.org/wiki/Apriori_algorithm). The final model takes as input a product of interest, and returns three recommendations of other products the consumer might be interested in.


## How does it work?

**Step 1:** Login to your UbiOps account at https://app.ubiops.com/ and create an API token with project editor
 rights. To do so, click on *Users & permissions* in the navigation panel, and then click on *API tokens*.
Click on *create token* to create a new token.

![Creating an API token](api_token_screenshot.png)

Give your new token a name, save the token in a safe place and assign the `project-editor` role on project level to it.

**Step 2:** Download the [recommender-system folder](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/cookbook/tree/master/recommender-system/recommender-system) and open `recommender.ipynb`. In the notebook you will find a space
to enter your API token and the name of your project in UbiOps. Paste the saved API token in the notebook in the indicated spot
and enter the name of the project in your UbiOps environment. This project name can be found in the top of your screen in the
WebApp. In the image in step 1 the project name is *scikit-example*.

**Step 3:** Run the Jupyter notebook `recommender.ipynb` and everything will be automatically deployed to your UbiOps environment! 
Afterwards you can explore the code in the notebook or explore the application in the UbiOps WebApp.
