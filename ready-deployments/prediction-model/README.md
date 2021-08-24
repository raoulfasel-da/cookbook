# Load & run a pre-trained (prediction) model

<div class="videoWrapper">

<iframe src="https://www.youtube.com/embed/K_Nvd01WB4c" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

</div>


Prediction models are a typical data science application. They are very straightforward to deploy on UbiOps
and in [this deployment package](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/cookbook/tree/master/ready-deployments/prediction-model/predictor_package) you can see how it works. We have put the `deployment.py` here as well
for your reference:

```python
import os
import pandas as pd
from tensorflow.keras.models import load_model

class Deployment:

    def __init__(self, base_directory, context):
        
        print("Initialising KNN model")
        model_file = os.path.join(base_directory, "tensorflow_model.h5")
        self.model = load_model(model_file)

    def request(self, data):
        # Loading in the data that was sent with the request.
        print('Loading data')
        input_data = pd.read_csv(data['data'])
        
        # Calling the model for a prediction
        print("Prediction being made")
        prediction = self.model.predict(input_data)

        # After the prediction is made you can perform additional processing steps as you please
        # We simply write the prediction to a csv
        print('Writing prediction to csv')
        pd.DataFrame(prediction).to_csv('prediction.csv', header = ['MPG'], index_label= 'index')
        
        return {
            "prediction": 'prediction.csv'
        }
```

In the `__init__` method of the Deployment class
we load in a trained model file. In the `request` method we call `model.predict` to actually make the 
prediction. This structure works for models created with Tensorflow, ScikitLearn or other standard data
science libraries. For more info on that please see [the Tensorflow](../../tensorflow-example/README.md) and 
[the Scikit-learn](../../scikit-deployment/README.md) examples. 


## Running the example in UbiOps

To deploy this example model to your own UbiOps environment you can log in to the WebApp and create a new 
deployment in the deployment tab. You will be prompted to fill in certain parameters, you can use the 
following:

| Deployment configuration | |
|--------------------|--------------|
| name | prediction-model|
| description | A standard prediction model|
| input field: | name = data, datatype = blob |
| output field: | name = prediction, datatype = blob |
| version name | v1 |
| description | leave blank |
| language | python 3.6 |
| upload code | [deployment zip](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/cookbook/tree/master/ready-deployments/prediction-model/predictor_package) _do not unzip!_|
| Deployment mode | Express |
| Request retention | Leave on default settings |

The advanced parameters and labels can be left as they are. They are optional.

After uploading the code and with that creating the deployment version UbiOps will start deploying. Once
you're deployment version is available you can make requests to it. [Here](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/cookbook/tree/master/ready-deployments/prediction-model/dummy_data_to_predict.csv) you can find some dummy data to
use as input for this model.
