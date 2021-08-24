import json
import ubiops

import azure.functions as func


PROJECT_NAME = '<YOUR PROJECT NAME>'
PIPELINE_NAME = 'example-pipeline'


def main(req: func.HttpRequest):
    """
    Deployment request that is HTTP Triggered.

    :param req: HttpRequest object
    """

    # Get the POST request body
    req_body = req.get_json()

    configuration = ubiops.Configuration()
    configuration.api_key['Authorization'] = 'Token <YOUR TOKEN HERE>'

    client = ubiops.ApiClient(configuration)
    api = ubiops.api.CoreApi(client)
    r = api.pipeline_requests_create(
        project_name=PROJECT_NAME,
        pipeline_name=PIPELINE_NAME,
        data={'data': json.dumps(req_body['value']), 'training': False}
    )

    return json.dumps({'output': f"Response of pipeline request is {r}"})
