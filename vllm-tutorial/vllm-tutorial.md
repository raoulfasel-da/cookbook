[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/vllm-tutorial/vllm-tutorial){ .md-button .md-button--primary } [View source code :fontawesome-brands-github:](https://github.com/UbiOps/tutorials/blob/master/vllm-tutorial/vllm-tutorial/vllm-tutorial.ipynb){ .md-button }


# Deploy a streaming LLM server on UbiOps with vLLM

In this tutorial, we will explain how to run your LLMs on UbiOps with [vLLM](https://github.com/vllm-project/vllm) by setting up a vLLM server in your deployment.
vLLM is an LLM serving framework that implements several techniques to increase model throughput, and allows a single LLM
to process multiple requests concurrently.

In our example, we spin up an [OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html?ref=blog.mozilla.ai). The request method of the deployment is used to route request data to the `v1/chat/completions` endpoint of this  vLLM server,
allowing for chatlike use cases.

The deployment accepts input of type `dict` in the OpenAI chat completion format, and returns output in the same standard.

For demo purposes, we will deploy a vLLM server that hosts the [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B). To follow along, ensure that 
your UbiOps subscription has access to GPUs, and that you have a Huggingface tokens with sufficient permissions to read out Llama v3.2 1B.


The following steps will be performed in this tutorial:

1. Set up a connection with UbiOps
3. Create a UbiOps deployment that deploys the server
4. Initiate a batch of requests to be handled by the deployment that hosts the vLLM server

## 1. Set up a connection with the UbiOps API client


First, we will need to install the UbiOps Python Client Library to interface with UbiOps from Python.


```python
!pip install -qU ubiops
```

Now, we will need to initialize all the necessary variables for the UbiOps deployment and the deployment directory,
which we will zip and upload to UbiOps.


```python
API_TOKEN = "<INSERT API TOKEN WITH PROJECT EDITOR RIGHTS>"
PROJECT_NAME = "<INSERT YOUR PROJECT NAME>"
DEPLOYMENT_NAME = "llama-server"
DEPLOYMENT_VERSION = "v1"  # Choose a name for the version.

HF_TOKEN = "<ENTER YOUR HF TOKEN WITH ACCESS TO A LLAMA REPO HERE>"  # We need this token to download the model from Huggingface 

print(f"Your new deployment will be called: {DEPLOYMENT_NAME}.")
```

At last, let's initialize the UbiOps client.


```python
import ubiops

configuration = ubiops.Configuration(host="https://api.ubiops.com/v2.1")
configuration.api_key["Authorization"] = API_TOKEN

client = ubiops.ApiClient(configuration)
api = ubiops.CoreApi(client)
api.service_status()
```

And let's create a deployment package directory, where will add our [deployment package files](https://ubiops.com/docs/deployments/deployment-package/deployment-structure/)


```python
import os

dir_name = "deployment_package"
os.makedirs(dir_name, exist_ok=True)
```

***


## 2. Setup deployment environment
In order to use vLLM inside the deployment, we need to set up the environment of the deployment so that everything will run smoothly.  
This will be done by specifying the `requirements.txt`.
More information on these files can be found in the [UbiOps docs](https://ubiops.com/docs/environments/#uploading-dependency-information)


All we need to do now is to create the `requirements.txt` file. Note that `vllm` automatically installs the CUDA drivers
that are required to load the underlying model on a GPU.


```python
%%writefile {dir_name}/requirements.txt
vllm
openai
```

## 3. Creating UbiOps deployment

In this section, we will create the UbiOps deployment. 
This will be done by creating the deployment code that will run on UbiOps. 
We will furthermore archive the deployment directory and upload it to UbiOps. 
This will create a deployment and a version of the deployment on UbiOps and make it available for use.

## 3.1 Creating deployment code

### Creating Deployment Code for UbiOps

We will now create the deployment code that will run on UbiOps. This involves creating a `deployment.py` file containing 
a `Deployment` class with two key methods:

- **`__init__` Method**  
  This method runs when the deployment starts. It can be used to load models, data artifacts, and other requirements for inference.

- **`request()` Method**  
  This method executes every time a call is made to the model's REST API endpoint. It contains the logic for processing incoming data.

We will configure [`instance_processes`](https://ubiops.com/docs/requests/request-concurrency/#request-concurrency-per-instance) to 10, 
allowing each deployment instance to handle 10 concurrent requests. The model will be loaded as a background process within the `__init__` 
of the first process. A client will also be initialized in each process to proxy requests from all running processes to the host LLM.

For a complete overview of the deployment code structure, refer to the [UbiOps documentation](https://ubiops.com/docs/deployments/deployment-package/deployment-structure/).



```python
%%writefile {dir_name}/deployment.py

import os
import subprocess
import logging
import time
import requests
import uuid
import json

from openai import OpenAI, BadRequestError
import torch

logging.basicConfig(level=logging.INFO)

class PublicError(Exception):
    # Ensure that any OpenAI specific error regarding e.g. exceeding max_model_len is propagated to the end user
    def __init__(self, public_error_message):
        super().__init__()
        self.public_error_message = public_error_message


class Deployment:
    def __init__(self, context):
        self.model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
        self.context = context
        self.model_length = os.environ.get("MAX_MODEL_LEN", 2048)
        self.vllm_gpu_memory_utilization = os.environ.get("GPU_MEMORY_UTILIZATION", 0.95)
        
        # Default token generation parameters 
        self.default_config = {
            'temperature': float(os.environ.get("TEMPERATURE_DEFAULT", 0.8)),
            'max_tokens' : int(os.environ.get("MAX_TOKENS_DEFAULT", 512))
        }

        # Start the vLLM server in the first process
        if int(context["process_id"]) == 0: 
            logging.info("Initializing vLLM")
            self.vllm_process = self.start_vllm_server()
            # Poll the health endpoint to ensure the server is ready before initiating clients.
            self.poll_health_endpoint()

        # In each process, set up a client that connects to the local vLLM server
        self.client = OpenAI(base_url="http://localhost:8000/v1", api_key="OPENAI_KEY")



    def request(self, data, context):
        """
        Method for deployment requests, called separately for each individual request.
        Integrates vLLM server request logic using the OpenAI client.
        See https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        and https://docs.vllm.ai/en/latest/quantization/auto_awq.html
        """
        print("Processing request")

        # Parse input prompt
        openai_chat_template = data["input"]

        messages = openai_chat_template["messages"]
        config = self.default_config.copy()
        config.update(openai_chat_template.get("config", {}))

        is_streaming = openai_chat_template.get("stream", False)  # default to non-streaming

        # Always add stream_options for usage stats if streaming is enabled
        if is_streaming:
            config["stream_options"] = {"include_usage": True}

        # Process the request
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **config,
                stream=is_streaming
            )
        except BadRequestError as e:
            raise PublicError(str(e))

        if is_streaming:
            streaming_callback = context["streaming_update"]
            response_text = ""

            for partial_response in response:
                print(f"partial response: {partial_response}")
                if partial_response.choices:
                    partial_text = partial_response.choices[0].delta.content
                    if partial_text:
                        streaming_callback(partial_text)
                        response_text += partial_text
                if partial_response.usage:
                    final_chunk = partial_response

            return {
                "output": {
                    "id": final_chunk.id,
                    "object": final_chunk.object,
                    "created": final_chunk.created,
                    "model": final_chunk.model,
                    "choices": [{"message": {"content": response_text}}],
                    "usage": final_chunk.usage.dict() if final_chunk.usage else None
                }
            }


        # Non-streaming response
        return {"output": response.dict()}


    def start_vllm_server(self):
        """
        Starts the vLLM server in a subprocess with the specified model.
        """

        self.vllm_path = find_executable("vllm")
        vllm_process = subprocess.Popen([self.vllm_path, 
                                         "serve", 
                                         self.model_name, 
                                         "--max_model_len",
                                         str(self.model_length),
                                         "--dtype",
                                         "half",
                                         "--gpu-memory-utilization",
                                         str(self.vllm_gpu_memory_utilization),
                                         "--tensor-parallel-size",  # Grab all GPUs available on the instance
                                         str(torch.cuda.device_count()),
                                         "--api-key",
                                         "OPENAI_KEY"]
        )

        logging.info("Starting vLLM server")
        return vllm_process

    def poll_health_endpoint(self):
        """
        Polls the /health endpoint to ensure the vLLM server is ready before processing requests.
        """
        logging.info("Initiating vLLM server. This can take a couple of minutes...")
        while True:
            poll = self.vllm_process.poll()
            # Ensure to
            if poll is not None:
                logging.error("vLLM server process terminated unexpectedly.")
                raise RuntimeError(f"vLLM server process exited with code: {poll}")

            try:
                resp = requests.get('http://localhost:8000/health', timeout=5)
                if resp.status_code == 200:
                    logging.info("vLLM server is ready")
                    break
                else:
                    logging.warning(f"Unexpected status code: {resp.status_code}. Retrying...")
            except requests.exceptions.ConnectionError:
                time.sleep(5)
            except requests.exceptions.RequestException as e:
                logging.error(f"Request failed: {e}")
                time.sleep(5)


def find_executable(executable_name):
    """
    Find the path to the executable in virtual environment or system paths.
    """
    path = subprocess.run(['which', executable_name], capture_output=True, text=True, check=True).stdout.strip()

    def is_executable(file_path):
        return os.path.isfile(file_path) and os.access(file_path, os.X_OK)

    if is_executable(path):
        logging.info(f"The path to the executable is: {path}")
        return path

```

### 3.2 Create UbiOps deployment

Now we can create the deployment, where we define the in- and outputs of the model.
Each deployment can have multiple versions. For each version you can deploy different code, environments, instance types etc.

We will use the following inputs and outputs in the deployment:

| Type   | Field Name    | Data Type |
|--------|---------------|-----------|
| Input  | input         | dict      |
| Output | output        | dict      |




```python
deployment = api.deployments_create(
    project_name=PROJECT_NAME,
    data={
        "name": DEPLOYMENT_NAME,
        "description": "a vLLM deployment",
        "input_type": "structured",
        "output_type": "structured",
        "input_fields": [
            {"name": "input", "data_type": "dict"},
        ],
        "output_fields": [
            {"name": "output", "data_type": "dict"},
        ]
    }
)
print(deployment)

```

### 3.3 Create a deployment version
Next we create a version for the deployment. For the version we set the name, environment and size of the instance (we're using a GPU instance type here, check if the instance type specified here is available!).


```python
version_template = {
    "version": DEPLOYMENT_VERSION,
    "environment": "python3-12",
    "instance_type_group_name": "16384 MB + 4 vCPU + NVIDIA Ada Lovelace L4",
    "maximum_instances": 1,
    "minimum_instances": 0,
    "instance_processes": 10,
    "maximum_idle_time": 900,
}

deployment_version = api.deployment_versions_create(
        project_name=PROJECT_NAME,
        deployment_name=DEPLOYMENT_NAME,
        data=version_template,
    )

print(deployment_version)
```

Here we create environment variables for the Huggingface token.
We need this token to allow us to download models from gated HuggingFace repos.
The standard model used in this deployment is `meta-llama/Meta-Llama-3.2-1B`.
This model is available in a gated HuggingFace repo, so we need to provide the token to access it.

If you want to use a different model,
you can change the deployment code or add an `MODEL_NAME` environment variable by using similar code as the code cell below:


```python
api_response = api.deployment_version_environment_variables_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    data=ubiops.EnvironmentVariableCreate(name="HF_TOKEN", value=HF_TOKEN, secret=True),
)
```

### 3.4 Archive deployment


```python
import shutil

# Archive the deployment directory
deployment_zip_path = shutil.make_archive(dir_name, 'zip', dir_name)
```

### 3.5 Upload deployment
We will now upload the deployment to UbiOps. In the background, This step will take some time, because UbiOps interprets
the environment files and builds a docker container out of it. You can check the UI for any progress.


```python
upload_response = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    file=deployment_zip_path,
)
print(upload_response)

# Check if the deployment is finished building. This can take a few minutes
ubiops.utils.wait_for_deployment_version(
    client=api.api_client,
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    revision_id=upload_response.revision,
)
```

## 4. Making requests to the deployment

Our deployment is now live on UbiOps! Let's test it out by sending a bunch of requests to it.
This request will be a simple prompt to the model, asking it to respond to aquestion.
In case your deployment still needs to scale, it may take some time before your first request is picked up. You can check
the logs of your deployment version to see if the vLLM server is ready to accept requests.

Let's first prepare the requests:


```python
import json

questions = [
    "What is the weather like today?",
    "How do I cook pasta?",
    "Can you explain quantum physics?",
    "What is the capital of France?",
    "How do I learn Python?"
]

requests_data = [
    {
        "input": {
            "config": {
                "max_tokens": 256,
                "temperature": 0.8
            },
            "messages": [
                {
                    "content": "You are a helpful assistant.",
                    "role": "system"
                },
                {
                    "content": question,
                    "role": "user"
                }
            ],
            "stream": False
        }
    }
    for question in questions
]

print(json.dumps(requests_data, indent=2))

```

And then create the requests:


```python

requests = api.batch_deployment_requests_create(
    project_name=PROJECT_NAME, deployment_name=DEPLOYMENT_NAME, data=requests_data, timeout=3600
)
```

And wait for them to complete:


```python
import time
request_ids = [request.id for request in requests]

while True:
    request_statuses = [request.status for request in api.deployment_requests_batch_get(PROJECT_NAME, DEPLOYMENT_NAME, request_ids )]
    if all(request_status == "completed" for request_status in request_statuses):
        print("All requests handled succesfully!")
        break
    time.sleep(1)

```

From the request start times, you can infer that all requests were processed simultaneously:


```python
request_start_times = [request.time_started for request in api.deployment_requests_batch_get(PROJECT_NAME, DEPLOYMENT_NAME, request_ids)]
print(request_start_times)
```

### Sending a request with streaming output

For this request, we will add the key `stream: true` to the input, enabling streaming responses


```python
streaming_request_data = {
    "input": {
        "config": {
            "max_tokens": 256,
            "temperature": 0.8
        },
        "messages": [
            {
                "content": "You are a helpful assistant.",
                "role": "system"
            },
            {
                "content": "Can you stream your response?",  
                "role": "user"
            }
        ],
        "stream": True 
    }
}

data = {"input": streaming_request_data["input"]}

# Create a streaming deployment request
for item in ubiops.utils.stream_deployment_request(
    client=api.api_client,
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    data=data,
    timeout=3600,
    full_response=False,
):
    print(item, end="")

```

That's it! Even though the model itself claims it does not stream, we still ensured it did.

We have set up a deployment that hosts a vLLM server. This tutorial just serves as an example. Feel free to reach out to
our [support portal](https://www.support.ubiops.com) if you want to discuss your set-up in more detail.

## 5. Cleanup

At last, let's close our connection to UbiOps


```python
client.close()
```
