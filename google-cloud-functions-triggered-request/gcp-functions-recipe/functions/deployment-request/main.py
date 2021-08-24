import base64
import ubiops


def ubiops_request(event, context):
    """
    Deployment request triggered from a message on a Cloud Pub/Sub topic.

    :param dict event: Event payload.
    :param google.cloud.functions.Context context: Metadata for the event.
    """

    pubsub_message = base64.b64decode(event['data']).decode('utf-8')

    # The API Token for UbiOps is hardcoded for simplicity in this example.
    # This should *absolutely never* be done in a production like environment.
    # Instead make use of the solutions provided, in this case by Google, to handle secrets and passwords.
    configuration = ubiops.Configuration()
    configuration.api_key['Authorization'] = 'Token abcdefghijklmnopqrstuvwxyz'

    client = ubiops.ApiClient(configuration)
    api = ubiops.api.CoreApi(client)
    api.deployment_requests_create(
        project_name='test-project',
        deployment_name='test-deployment',
        version='version',
        data=pubsub_message
    )
