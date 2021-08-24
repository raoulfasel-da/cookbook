# Age estimation with a Mendix front end
This deployment is made for running a neural network model developed with Caffe. Be aware that the model weight file is not included and needs to be downloaded separately before uploading this deployment package to UbiOps. See the download link below.

This deployment is part of an article: 
[Building a low code app powered by AI](https://ubiops.com/building-a-low-code-app-powered-by-ai/)

Please read the article for more information on how this model was used in practice in the background of a low-code Mendix app.

_Download link for deployment package_: [caffe-deployment](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/cookbook/tree/master/mendix-age-estimation/deployment_package_caffe)

_Download link for weight files_: [weight-files](https://storage.googleapis.com/ubiops/data/dex_chalearn_iccv2015.caffemodel)

Add these weight files to the deployment package before uploading.

## Running the example in UbiOps

Please take a look at the article for more information:
[Building a low code app powered by AI](https://ubiops.com/building-a-low-code-app-powered-by-ai/)

| Deployment configuration | |
|--------------------|--------------|
| name | age-estimator|
| description | A model for estimating age|
| input field: | name = photo, datatype = string (base64 encoded) |
| output field: | name = age, datatype = int |
| version name | v1 |
| description | leave blank |
| language | python 3.6 |
