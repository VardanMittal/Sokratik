from transformers import pipeline
import torch

# This is the model we will fine-tune later
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
prompt = "I'm feeling very anxious about an interview tomorrow. What should I do?"

# Setup the text generation pipeline
generator = pipeline(
    "text-generation",
    model=model_name,
    torch_dtype=torch.bfloat16, # Use bfloat16 for efficiency
    device_map="auto" # This will use your GPU if you have one
)

# Generate the response
response = generator(prompt, max_new_tokens=250)

print("--- BASELINE MODEL RESPONSE ---")
print(response[0]['generated_text'])

# Save the response
with open("baseline_response.txt", "w") as f:
    f.write(response[0]['generated_text'])