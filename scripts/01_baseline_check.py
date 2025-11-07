import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# The model ID for Llama 3
model_id = "meta-llama/Llama-3-8B-Instruct"

# This is the 4-bit quantization configuration
# It tells transformers to load the model in 4-bits
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Loading tokenizer...")
# The tokenizer doesn't need the 4-bit config
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Loading model... (This may take a few minutes)")
# Load the model with the 4-bit config
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto", # This will automatically use the T4 GPU
    torch_dtype=torch.bfloat16,
)

# --- Run the Baseline Check ---
prompt_text = "I'm feeling very anxious about an interview tomorrow. What should I do?"

# Llama 3 uses a specific chat template.
# We must use it, or the model's answers will be low-quality.
chat = [
    # We add a system prompt to set the context
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt_text},
]

# This helper function formats the chat into the correct Llama 3 prompt string
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("\nGenerating response...")

# Generate the response
# We add terminators to tell Llama 3 when to stop
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    **inputs,
    max_new_tokens=250,
    eos_token_id=terminators,
    do_sample=True, # Allows for more creative answers
    temperature=0.7, 
    top_p=0.9,
)

# Decode and print the response
# The response includes the prompt, so we slice it
response_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
response_answer = response_full[len(prompt):] # Get just the model's answer

print("\n--- BASELINE MODEL RESPONSE (Llama 3 8B) ---")
print(response_answer)

# Save the response
with open("baseline_response_llama3.txt", "w") as f:
    f.write("--- PROMPT ---\n")
    f.write(prompt_text)
    f.write("\n\n--- LLAMA 3 8B RESPONSE ---\n")
    f.write(response_answer)

print("\nBaseline response saved to 'baseline_response_llama3.txt'")