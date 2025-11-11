import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# --- CONFIG ---
# The base model
BASE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# --- CHANGE #1: Point to your new V2 model ---
ADAPTER_ID = "vardan10/Sokratik-v2"
# YOUR new model on the Hub. 

# The same prompt from our baseline check
PROMPT = "I'm feeling very anxious about an interview about my life. What should I do?"

# --- 1. Load Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

# --- 2. Load Base Model (Quantized) ---
print("Loading base model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
    # llm_int8_enable_fp32_cpu_offload=True  <-- THIS LINE IS REMOVED
)

# --- 3. Load Your Fine-Tuned Adapter ---
print(f"Loading Stoic adapter: {ADAPTER_ID}...")
model = PeftModel.from_pretrained(model, ADAPTER_ID)
model.eval() # Set model to evaluation mode
print("Adapter loaded!")

# --- 4. Format the Prompt (Llama 3 Instruct format) ---
chat = [
    {"role": "system", "content": "You are a Stoic philosopher. Answer with wisdom, logic, and tranquility."},
    {"role": "user", "content": PROMPT},
]

prompt_formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt_formatted, return_tensors="pt").to("cuda")

# --- 5. Generate Response ---
print("\n--- Generating Stoic Response ---")
outputs = model.generate(
    **inputs,
    max_new_tokens=250,
    eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

response_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
# Get just the model's answer
response_answer = response_full.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1]

print(f"\nPROMPT:\n{PROMPT}")
print("\nSTOIC RESPONSE:\n")
print(response_answer)

# Save to a new output file ---
with open("stoic_response_v2.txt", "w") as f:
    f.write(f"PROMPT:\n{PROMPT}\n\nSTOIC RESPONSE:\n{response_answer}")

print("\n\nResponse saved to 'stoic_response_v2.txt'")