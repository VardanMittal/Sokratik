import json 
import os

#Paths
BASE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
INPUT_CORPUS = os.path.join(BASE_FILE, "data/processed/stoic_corpus.txt")
OUTPUT_JSONL = os.path.join(BASE_FILE, "data/final/train.jsonl")

# Config
MIN_LENGTH = 50 # this helps in skipping very short context

def format_to_jsonl(corpus_path, output_path):
    print(f"Reading from {corpus_path}...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        full_text = f.read()

    # Split by double newlines to get paragraphs/sections
    # This assumes your cleaning script used \n\n to separate ideas.
    chunks = full_text.split('\n\n')
    
    print(f"Found {len(chunks)} raw chunks. Filtering and formatting...")

    count = 0
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for chunk in chunks:
            clean_chunk = chunk.strip()
            # Only keep chunks that are substantial enough to learn from
            if len(clean_chunk) >= MIN_LENGTH:
                # Create the JSON object.
                # For now, simple {"text": ...} format is perfect for learning style.
                data_object = {"text": clean_chunk}
                
                # Write it as a single line
                f_out.write(json.dumps(data_object, ensure_ascii=False) + "\n")
                count += 1

    print(f"--- SUCCESS ---")
    print(f"Created {output_path} with {count} training examples.")

# --- Main Execution ---
os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
format_to_jsonl(INPUT_CORPUS, OUTPUT_JSONL)