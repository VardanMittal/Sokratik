import os
import re


BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

RAW_DIR = os.path.join(BASE_DIR, "data/raw")
PROCESSED_PATH = os.path.join(BASE_DIR, "data/processed/stoic_corpus.txt")

def clean_gutenberg_text(text):
    """
    Aggressively cleans Project Gutenberg texts.
    """
    # 1. Identify Start/End Markers common in Gutenberg texts
    # Look at your actual files to confirm these phrases!
    start_markers = ["*** START OF", "THE FIRST BOOK", "BOOK I"]
    end_markers = ["*** END OF", "End of the Project Gutenberg", "INDEX"]

    start_idx = 0
    end_idx = len(text)

    for marker in start_markers:
        if marker in text:
             # Find marker, move past it to the next newline
            idx = text.find(marker)
            if idx != -1:
                 # Rough heuristic: jump 500 chars to skip titles/intros
                 # Adjust this based on manual inspection!
                start_idx = max(start_idx, idx + 500)

    for marker in end_markers:
        if marker in text:
            idx = text.find(marker)
            if idx != -1:
                end_idx = min(end_idx, idx)

    text = text[start_idx:end_idx]

    # 2. Remove specific junk using Regex
    # Remove "BOOK I", "CHAPTER V", etc.
    text = re.sub(r'^(BOOK|CHAPTER|LETTER)\s+[IVXLC\d]+.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    # Remove verse numbers like [1], [2] or 1., 2.
    text = re.sub(r'^\[?\d+\]?\.?', '', text, flags=re.MULTILINE)
    # Remove text inside brackets [like this translator note]
    text = re.sub(r'\[.*?\]', '', text)

    # 3. Normalize whitespace
    # Replace multiple newlines with exactly two (paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

# --- Main Execution ---
# Ensure processed directory exists
os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)

full_corpus = ""

for filename in os.listdir(RAW_DIR):
    if filename.endswith(".txt"):
        filepath = os.path.join(RAW_DIR, filename)
        print(f"Processing {filename}...")
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            raw_data = f.read()
            cleaned_data = clean_gutenberg_text(raw_data)
            full_corpus += cleaned_data + "\n\n" # Add space between books

# Save the unified corpus
with open(PROCESSED_PATH, 'w', encoding='utf-8') as f:
    f.write(full_corpus)

print(f"--- DONE! ---\nSaved unified corpus to: {PROCESSED_PATH}")
print(f"Total characters: {len(full_corpus)}")