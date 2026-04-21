import boto3
import json
import base64
bedrock=boto3.client(service_name="bedrock-runtime",region_name="us-east-1")
payload = {
    "textToImageParams": {
        "text": "Create image of a BMW bike rider on a highway, realistic style"
    },
    "taskType": "TEXT_IMAGE",
    "imageGenerationConfig": {
        "cfgScale": 8,
        "seed": 0,
        "width": 1024,
        "height": 1024,
        "numberOfImages": 3
    }
}
response=bedrock.invoke_model(
    modelId="amazon.titan-image-generator-v2:0",
    body=json.dumps(payload)
)


# ✅ Parse response
result = json.loads(response["body"].read())

# ✅ Extract base64 image
image_base64 = result["images"][0]

# ✅ Convert to binary
image_bytes = base64.b64decode(image_base64)

# ✅ Save as file
with open("output.png", "wb") as f:
    f.write(image_bytes)

print("Image saved as output.png")