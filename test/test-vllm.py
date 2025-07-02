import litellm

# Set litellm to verbose mode for more detailed output
litellm.set_verbose=True

try:
    response = litellm.completion(
        model="hosted_vllm/shisa-v2-mistral-small-24b",
        messages=[{"role": "user", "content": "Hi, how are you?"}],
        # Make sure this is the correct address and port for your vLLM server
        api_base="http://localhost:1234/v1"  # Or your vLLM server's address
    )
    print(response)
except Exception as e:
    print(f"An error occurred: {e}")