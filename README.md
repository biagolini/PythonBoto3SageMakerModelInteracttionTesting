# AWS SageMaker Flan-T5 XL Model Testing

This repository contains a script to test the Flan-T5 XL model (https://huggingface.co/google/flan-t5-xl) on AWS SageMaker. The script sets up a SageMaker endpoint and sends sample queries to the model.

## Setup

1. **AWS Configuration**: Ensure you have AWS credentials configured. You can use the AWS CLI to configure your credentials:

   ```sh
   aws configure
   ```

2. **Dependencies**: Install the required Python packages.
   ```sh
   pip install boto3
   ```

## Usage

1. **Script Execution**: The script `main.py` does the following:

   - Creates a session with configured AWS credentials.
   - Lists available SageMaker endpoints.
   - Uses the first endpoint in the list to send sample queries.
   - Prints the input and generated text for each query.

2. **Running the Script**:
   ```sh
   python main.py
   ```

## Parameter Explanation

In the `query_endpoint` function, you can customize the query parameters to control the behavior of the model:

- max_new_tokens: Specifies the maximum number of new tokens to generate.
- top_k: Limits the next token selection to the top-k most likely tokens.
- op_p: Uses nucleus sampling to select tokens with a cumulative probability above a threshold.
- do_sample: Enables random sampling of the tokens.
  For more details on these parameters, you can refer to the [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationMixin.generate).

## Testing

This project was tested using Amazon IVS with a channel configured for low-latency. This setup is intended to provide a real-time streaming experience with minimal delay, suitable for scenarios where interaction and immediacy are critical.

## Contributing

Feel free to submit issues, create pull requests, or fork the repository to help improve the project.

## License and Disclaimer

This project is open-source and available under the MIT License. You are free to copy, modify, and use the project as you wish. However, any responsibility for the use of the code is solely yours. Please use it at your own risk and discretion.
