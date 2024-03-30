import boto3
import json
from core import model

'''
Here we build the layers from s3
'''
# globals
client = boto3.client('s3')

# skeleton code (for weights)
weights = client.getS3()

#Nova transformer with feed forward
