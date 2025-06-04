"""Retrieve the CodeArtifact repository URL for the Python package index.

During the pre-install steps of .readthedocs.yaml configuration,
the URL is printed to be set in the pip configuration file.
This enables the install steps access to the CodeArtifact repository.
"""

import boto3

# AWS Configuration
AWS_REGION = 'us-east-1'
DOMAIN = 'cellarity'
DOMAIN_OWNER = '460684206204'

# Get Token
client = boto3.client('codeartifact', region_name=AWS_REGION)
token_response = client.get_authorization_token(domain=DOMAIN, domainOwner=DOMAIN_OWNER)

codeartifact_token = token_response['authorizationToken']
repository_url = f'https://aws:{codeartifact_token}@cellarity-460684206204.d.codeartifact.us-east-1.amazonaws.com/pypi/python/simple/'

# Print the URL for exporting
print(repository_url)
