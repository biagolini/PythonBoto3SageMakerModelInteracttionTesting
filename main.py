import boto3
import json

# Create a session with configured AWS credentials
session = boto3.Session()

# Obtain the SageMaker client
sagemaker_client = session.client('sagemaker')

# List all inference components
sagemaker_interface_components = sagemaker_client.list_inference_components()

# Retrieve the endpoint and inference component names
endpoint_name = sagemaker_interface_components['InferenceComponents'][0]['EndpointName']
inference_component_name = sagemaker_interface_components['InferenceComponents'][0]['InferenceComponentName']

# Check if there are any available endpoints
if not inference_component_name:
    print("No interface component found.")
else:
    print(f"Using endpoint: {endpoint_name}")
    print(f"Using interface: {inference_component_name}")

def query_endpoint(input_text, max_new_tokens=50, top_k=None, top_p=None, do_sample=False):
    """
    Function to query the SageMaker endpoint with provided parameters.
    """
    sagemaker_runtime_client = boto3.client("runtime.sagemaker")
    payload = {
        "inputs": input_text,
        "parameters": {
            "max_new_tokens": max_new_tokens,
        }
    }
    
    # Add optional parameters if they are provided
    if top_k is not None:
        payload["parameters"]["top_k"] = top_k
    if top_p is not None:
        payload["parameters"]["top_p"] = top_p
    if do_sample:
        payload["parameters"]["do_sample"] = do_sample

    # Invoke the SageMaker endpoint with the payload
    response = sagemaker_runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        InferenceComponentName=inference_component_name,
        ContentType="application/json",
        Body=json.dumps(payload).encode("utf-8")
    )
    
    # Parse and print the model's predictions
    model_predictions = json.loads(response["Body"].read())
    generated_text = model_predictions[0]["generated_text"]
    print(f"Input Text: {input_text}\nGenerated Text: {generated_text}\n")

# Define questions to query the endpoint
question1 = "What is AWS Lambda?"
question2 = "Explain the benefits of using Amazon S3."
question3 = "How can I secure my AWS infrastructure?"

# Ask questions and print results
query_endpoint(question1)
query_endpoint(question2, max_new_tokens=50)
query_endpoint(question3, max_new_tokens=200, top_k=50, top_p=0.95, do_sample=True)
