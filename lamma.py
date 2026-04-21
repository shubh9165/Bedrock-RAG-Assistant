import json
import boto3

bedrock=boto3.client(service_name="bedrock-runtime",region_name="us-east-1")

prompt="""

"""
response=bedrock.converse(
    modelId="meta.llama3-8b-instruct-v1:0",
    messages=[
        {
            "role":"user",
            "content":[{"text":"act as poem writer and write the poam on genrative ai"}]
        }
    ],
    inferenceConfig={
        "maxTokens":512,
        "temperature":0.5,
        "topP":0.9
    }
)





print(response)