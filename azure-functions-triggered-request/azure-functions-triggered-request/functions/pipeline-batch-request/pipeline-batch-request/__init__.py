import ubiops

import azure.functions as func


PROJECT_NAME = '<YOUR PROJECT NAME>'
PIPELINE_NAME = 'example-pipeline'
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
    r = api.batch_pipeline_requests_create(
        project_name=PROJECT_NAME,
        pipeline_name=PIPELINE_NAME,
        data=req_body
    )

    return f"Response of pipeline batch request is {r}"
