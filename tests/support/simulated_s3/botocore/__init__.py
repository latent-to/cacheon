"""Simulated botocore for the simulated boto3 client: its config type and client error."""


class ClientError(Exception):
    """The SDK's service error: the code a caller reads is ``response["Error"]["Code"]``."""

    def __init__(self, error_response, operation_name):
        super().__init__(f"{operation_name}: {error_response['Error']['Code']}")
        self.response = error_response
