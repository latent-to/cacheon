"""Simulated ``botocore.config``: the client options the production opener passes."""


class Config:
    def __init__(self, **options):
        self.options = options
