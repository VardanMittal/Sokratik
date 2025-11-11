import json
import os
import random

BASE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
INPUT_CORPUS = os.path.join(BASE_FILE,"data/processed/stoic_corpus.txt")
OUTPUT_JSONL = os.path.join(BASE_FILE,"data/final/train_chat.jsonl")
MIN_LENGTH = 50

# We will use a system prompt and a few user prompts to make the model robust
SYSTEM_PROMPT = "You are a Stoic philosopher. Answer with wisdom, logic, and tranquility, drawing upon the principles of Stoicism."

# A list of user prompts to randomly pair with the Stoic texts
USER_PROMPTS = [
    "What is your wisdom on this?",
    "Please share your reflections.",
    "How should one think about this?",
    "Give me your philosophical insight.",
    "Reflect on this matter."
]

def format_to_chat_jsonl(corpus_path, output_path):
    print(f"Reading from {corpus_path}...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        full_text = f.read()

    chunks = full_text.split('\n\n')
    print(f"Found {len(chunks)} raw chunks. Filtering and formatting...")
    
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for chunk in chunks:
            clean_chunk = chunk.strip()
            if len(clean_chunk) >= MIN_LENGTH:
                
                # Select a random user prompt
                user_prompt = random.choice(USER_PROMPTS)
                
                # Create the Llama 3 "messages" format
                data_object = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": clean_chunk}
                    ]
                }
                
                f_out.write(json.dumps(data_object, ensure_ascii=False) + "\n")
                count += 1

    print(f"--- SUCCESS ---")
    print(f"Created {output_path} with {count} chat examples.")

# --- Main Execution ---
os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
format_to_chat_jsonl(INPUT_CORPUS, OUTPUT_JSONL)