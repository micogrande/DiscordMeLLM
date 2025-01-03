from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Path to save the base model
base_model_path = "../models/base_model"

# Specify the model name (note the -hf at the end)
model_name = "meta-llama/Llama-2-7b-chat-hf"

# Get token from environment variable or specify directly
token = "your_token_here"  # Replace with your actual token

# Create the directory if it doesn't exist
os.makedirs(base_model_path, exist_ok=True)

# Download the base model and tokenizer with token authentication
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=token,
    device_map="auto"
)

# Save the model locally
model.save_pretrained(base_model_path)
tokenizer.save_pretrained(base_model_path)

print(f"Base model downloaded and saved to {base_model_path}")