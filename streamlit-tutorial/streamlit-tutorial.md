# Streamlit & port forwarding tutorial

Note: This notebook runs on Python 3.11 and uses UbiOps Client Library 3.15.0.

In this notebook we will show you the following:

1. How to set-up a Streamlitdashboard that creates an interface for deployments inside your project.
2. How to host that dashboard locally.
3. How to run the dashboard in a deployment in UbiOps using the [port forwarding functionality](https://ubiops.com/docs/deployments/deployment-versions/#opening-up-a-port-from-your-deployment-beta).

To set-up this workflow, we will create two deployments:

- `image-recognition`: which will host a model that predicts hand written digits.
- `streamlit-host`: which will host a Streamlit dashboard using the 'port forwarding' functionality.

We will also create two Streamlit dashboards in this notebook, the first dashboard will be connected to the `image-recognition`
deployment and hosted locally. The second Streamlit dashboard will be hosted on the `streamlit-host` deployment, by 
spinning up a streamlitserver in the deployment, and opening up it's relevant port using the port-forwarding functionality.

Be aware that you cannot run everything in one go, because halfway a streamlit server is spun up on localhost.

To interface with UbiOps through code we need the UbiOps Python client library. In the following cell it will be installed.


```python
!pip install -qU UbiOps
```

## Establishing a connection with your UbiOps environment

Add your API token and project name. We provide a deployment and deployment version name. Afterwards we connect to the UbiOps API via our Python Client. This way we can deploy the MNIST model to your environment.


```python
import ubiops

API_TOKEN = "<INSERT API_TOKEN WITH PROJECT EDITOR RIGHTS>"  # Make sure this is in the format "Token token-code"
PROJECT_NAME = "<INSERT PROJECT NAME IN YOUR ACCOUNT>"


DEPLOYMENT_NAME = "image-recognition"
DEPLOYMENT_VERSION = "v1"

# Initialize client library
configuration = ubiops.Configuration(host="https://api.ubiops.com/v2.1")
configuration.api_key["Authorization"] = API_TOKEN

# Establish a connection
client = ubiops.ApiClient(configuration)
api = ubiops.CoreApi(client)
print(api.projects_get(PROJECT_NAME))
```

## Download the MNIST image recognition model 

For this tutorial we have prepared a basic deployment package that we can integrate into our Streamlit dashboard. Let us download the [deployment package](https://ubiops.com/docs/deployments/deployment-package/deployment-structure/) which you can also find in our 
[Git Repository](https://github.com/UbiOps/tutorials/tree/master/ready-deployments/image-recognition/mnist_deployment_package): 


```python
!curl -OJ "https://storage.googleapis.com/ubiops/example-deployment-packages/mnist_deployment_package.zip"
```

## 1. How to set-up a Streamlitdashboard that creates an interface for deployments inside your project.

### Create the deployment

Create a deployment. Here we define the in- and outputs of a model. We can create different deployment versions. For this 
deployment we will use the following configuration:

|Deployment input & output variables| **Variable name** |**Data type** |
|--------------------|--------------|--------------|
| **Input fields**   | image | file |
| **Output fields**  | prediction | integer |
|                    | probability        | double precision |


```python
# Create the deployment
deployment_template = ubiops.DeploymentCreate(
    name=DEPLOYMENT_NAME,
    description="image-recognition",
    input_type="structured",
    output_type="structured",
    input_fields=[
        {"name": "image", "data_type": "file"},
    ],
    output_fields=[
        {"name": "prediction", "data_type": "int"},
        {"name": "probability", "data_type": "double"},
    ],
    labels={"demo": "MNIST-Streamlit"},
)

api.deployments_create(project_name=PROJECT_NAME, data=deployment_template)
```

### Create a deployment version

Now we will create a version of the deployment. For the version we need to define the name, the environment, the type of instance (CPU or GPU) as well the size of the instance.


```python
# Create the version
version_template = ubiops.DeploymentVersionCreate(
    version=DEPLOYMENT_VERSION,
    environment="python3-11",
    instance_type_group_name="512 MB + 0.125 vCPU",
    minimum_instances=0,
    maximum_instances=1,
    maximum_idle_time=1800,  # = 30 minutes
    request_retention_mode="full",
)

api.deployment_versions_create(
    project_name=PROJECT_NAME, deployment_name=DEPLOYMENT_NAME, data=version_template
)
```


#### Upload the deployment package to UbiOps


```python
import shutil

deployment_code_archive = shutil.make_archive(
    "mnist_deployment_package", "zip", "mnist_deployment_package"
)

# Upload the zipped deployment package
file_upload_result = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    file="mnist_deployment_package.zip",
)

ubiops.utils.wait_for_deployment_version(
    client=api.api_client,
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    revision_id=file_upload_result.revision,
)
```

## 2. How to host that dashboard locally.

Enter your API token and Project in the Streamlit file. Then download the Streamlit package


```python
%%writefile mnist-streamlit.py
import streamlit as st
import ubiops 
import tempfile
from time import sleep
from ubiops import utils

st.title("Streamlit and UbiOps example")

# Connect with your UbiOps environment
API_TOKEN = '<INSERT API_TOKEN WITH PROJECT EDITOR RIGHTS>' # Make sure this is in the format "Token token-code"
PROJECT_NAME = '<INSERT PROJECT NAME IN YOUR ACCOUNT>'

DEPLOYMENT_NAME = 'image-recognition'

# API setup 
if PROJECT_NAME and API_TOKEN and DEPLOYMENT_NAME:
    # Only reconnect if API object is not in session state
    if 'ubiops_api' not in st.session_state:
        with st.spinner("Connecting to UbiOps API"):
            configuration = ubiops.Configuration(host="https://api.ubiops.com/v2.1")
            configuration.api_key['Authorization'] = API_TOKEN

            st.session_state.client = ubiops.ApiClient(configuration)
            st.session_state.ubiops_api = ubiops.CoreApi(st.session_state.client)
            deployment_info = st.session_state.ubiops_api.deployments_get(PROJECT_NAME,DEPLOYMENT_NAME)
           
            print(deployment_info)
            
            sleep(2) # sleep for 2s to showcase progress spinners
            
            # Use the streamlit session to store API object
            if(st.session_state.ubiops_api.service_status().status == 'ok' ):
                st.success("Connected to UbiOps API!")
            else:
                st.error("Not connected!")
                


# File upload
upload_file = st.file_uploader("Choose a file")
if upload_file is not None:
    if 'results' not in st.session_state:
        st.session_state.results = []
    with open("out.txt", "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(upload_file.getvalue())
    file_uri = ubiops.utils.upload_file(st.session_state.client, PROJECT_NAME, 'out.txt')
    # Make a request using the file URI as input.
    data = {'image': file_uri}
    
    result = st.session_state.ubiops_api.deployment_requests_create(
        project_name=PROJECT_NAME,
        deployment_name=DEPLOYMENT_NAME,
        data=data
    )
    # Store results in session
    st.session_state.results.append([result,upload_file])

# Show all results in from session
if 'results' in st.session_state: 
    for r in st.session_state.results[::-1]:
        c1, c2 = st.columns(2)
        c2.write(r[0].result)
        c1.image(r[1])

```

Install the Streamlit package


```python
!pip install streamlit
```

Now it is time to connect to spin up the Streamlitdashboard on our localhost:


```python
!streamlit run mnist-streamlit.py
```

You can download example request data by [clicking here](https://storage.googleapis.com/ubiops/data/Integration%20with%20other%20tools/streamlit-integration/streamlit-integration-request.zip)

## 3. How to run the dashboard in a deployment in UbiOps using the port forwarding functionality.

It is also possible to host your Streamlit dashboard on a UbiOps deployment. so that you expose the dashboard to the public. 
We can run Streamlit dashboards on UbiOps by means of port forwarding. Note that not every instance type available on UbiOps has port forwarding enabled.

In order to enable port forwarding we have to alter the `deployment.py` a bit, we also need to at streamlit to a custom
environment. Let's first create a new deployment package, and then push it to UbiOps.


```python
%mkdir deployment_package_streamlit
```


```python
import subprocess
import urllib.request
import uuid


class UbiOpsError(Exception):
    def __init__(self, error_message):
        super().__init__()
        self.public_error_message = error_message


class Deployment:
    def __init__(self):
        token = str(uuid.uuid4())

        self.proc = None
        self.port = "8888"

        print("Starting up Streamlitt app")
        try:
            self.proc = subprocess.Popen(
                [
                    "streamlit",
                    "run",
                    "streamlit-app.py",
                    "--server.address",
                    "0.0.0.0",
                    "--server.port",
                    self.port,
                    "--browser.gatherUsageStats",
                    "false",
                ]
            )
            outs, errs = self.proc.communicate(timeout=10)
            print(outs, errs)

        except FileNotFoundError:
            raise UbiOpsError("Unable to start streamlit: streamlit is unknown")

        except subprocess.TimeoutExpired:
            print(
                "Streamlit continues running in the background. No more logs from now on..."
            )

        # Get the IP address and print to the logs
        http_request = urllib.request.urlopen("http://whatismyip.akamai.com")
        self.ip_address = http_request.read().decode("utf8")
        http_request.close()

        print(f"The IP address of this deployment is: {self.ip_address}")

        self.dashboard_url = f"http://{self.ip_address}:8888/tree?token={token}"
        print(f"Dashboard URL: {self.dashboard_url}")

    def request(self, data):
        return {
            "ip_address": self.ip_address,
            "dashboard_url": self.dashboard_url,
            "port": int(self.port),
        }

    def stop(self):
        if self.proc is not None:
            self.proc.kill()
```


```python
%%writefile deployment_package_streamlit/requirements.txt
streamlit==1.32.1
ubiops==4.3.0
```

#### We now have our deployment package. Let's create a deployment and upload the package as a revision to it.

Note that the deployment above handles the image in the same manner as the deployment we created earlier, but for demonstration
purposes we now also return the `ip_address` and `port`.


```python
STREAMLIT_DEPLOYMENT_NAME = "streamlit-host"
STREAMLIT_VERSION_NAME = "v1"
```


```python
# Create the deployment
deployment_template = ubiops.DeploymentCreate(
    name=STREAMLIT_DEPLOYMENT_NAME,
    description="image-recognition",
    input_type="plain",
    output_type="structured",
    output_fields=[
        {"name": "ip_address", "data_type": "string"},
        {"name": "dashboard_url", "data_type": "string"},
        {"name": "port", "data_type": "int"},
    ],
    labels={"demo": "streamlit-hosting"},
)

api.deployments_create(project_name=PROJECT_NAME, data=deployment_template)
```

### Create the version


```python
# Create the version
version_template = ubiops.DeploymentVersionCreate(
    version=STREAMLIT_VERSION_NAME,
    environment="python3-11",
    instance_type_group_name="8192 MB + 2 vCPU (Dedicated)",
    minimum_instances=0,
    maximum_instances=1,
    maximum_idle_time=30,  # = 30 minutes
    request_retention_mode="full",
    ports=[{"public_port": 8888, "deployment_port": 8888, "protocol": "tcp"}],
)

api.deployment_versions_create(
    project_name=PROJECT_NAME,
    deployment_name=STREAMLIT_DEPLOYMENT_NAME,
    data=version_template,
)
```


```python
import shutil

shutil.make_archive(
    "deployment_package_streamlit", "zip", "deployment_package_streamlit"
)
```


```python
# Upload the zipped deployment package
file_upload_result = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=STREAMLIT_DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    file="deployment_package_streamlit.zip",
)

ubiops.utils.wait_for_deployment_version(
    client=api.api_client,
    project_name=PROJECT_NAME,
    deployment_name=STREAMLIT_DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    revision_id=file_upload_result.revision,
)
```




```python
%%writefile deployment_package_streamlit/streamlit-app.py

from time import sleep

import streamlit as st
import ubiops
from ubiops import utils

st.title("Streamlit and UbiOps example")

# Connect with your UbiOps environment
API_TOKEN = '<INSERT API_TOKEN WITH PROJECT EDITOR RIGHTS>' # Make sure this is in the format "Token token-code"
PROJECT_NAME = '<INSERT PROJECT NAME IN YOUR ACCOUNT>'
DEPLOYMENT_NAME = 'image-recognition'

# API setup 
if PROJECT_NAME and API_TOKEN and DEPLOYMENT_NAME:
    # Only reconnect if API object is not in session state
    if 'ubiops_api' not in st.session_state:
        with st.spinner("Connecting to UbiOps API"):
            configuration = ubiops.Configuration(host="https://api.ubiops.com/v2.1")
            configuration.api_key['Authorization'] = API_TOKEN

            st.session_state.client = ubiops.ApiClient(configuration)
            st.session_state.ubiops_api = ubiops.CoreApi(st.session_state.client)
            deployment_info = st.session_state.ubiops_api.deployments_get(PROJECT_NAME,DEPLOYMENT_NAME)
           
            print(deployment_info)
            
            sleep(2) # sleep for 2s to showcase progress spinners
            
            # Use the streamlit session to store API object
            if(st.session_state.ubiops_api.service_status().status == 'ok' ):
                st.success("Connected to UbiOps API!")
            else:
                st.error("Not connected!")

# File upload
upload_file = st.file_uploader("Choose a file")
if upload_file is not None:
    if 'results' not in st.session_state:
        st.session_state.results = []
    with open("out.txt", "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(upload_file.getvalue())
    file_uri = ubiops.utils.upload_file(st.session_state.client, PROJECT_NAME, 'out.txt')
    
    # Make a request using the file URI as input.
    result = st.session_state.ubiops_api.deployment_requests_create(
        project_name=PROJECT_NAME,
        deployment_name=DEPLOYMENT_NAME,
        data={'image': file_uri}
    )
    # Store results in session
    st.session_state.results.append([result,upload_file])

    # Store results in session
    st.session_state.results.append([result, upload_file])

# Show all results in from session
if 'results' in st.session_state:
    for r in st.session_state.results[::-1]:
        c1, c2 = st.columns(2)
        c2.write(r[0].result)
        c1.image(r[1])
```


```python
!streamlit run deployment_package_streamlit/streamlit-app.py
```

So that's it! You have just created two deployments, and connected the two of them using port forwarding!
