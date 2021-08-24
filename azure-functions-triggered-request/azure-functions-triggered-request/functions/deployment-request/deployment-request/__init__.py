import ubiops

import azure.functions as func


PROJECT_NAME = '<YOUR PROJECT NAME>'
DEPLOYMENT_NAME = 'example-deployment'
VERSION = 'v1'
TOKEN = '<YOUR TOKEN HERE>'


def main(req: func.HttpRequest):
    """
    Deployment request that is HTTP Triggered.

    :param req: HttpRequest object
    """

    # Get the POST request body
    req_body = req.get_json()

    configuration = ubiops.Configuration()
    configuration.api_key['Authorization'] = f"Token {TOKEN}"

    client = ubiops.ApiClient(configuration)
    api = ubiops.api.CoreApi(client)
    r = api.deployment_requests_create(
        project_name=PROJECT_NAME,
        deployment_name=DEPLOYMENT_NAME,
        version=VERSION,
        data=req_body
    )

    return f"Response of deployment request is {r}"
