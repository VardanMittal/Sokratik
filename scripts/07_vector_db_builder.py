import os
import shutil
from langchain_community.document_loaders import TextLoader
# --- FIX: Update import for newer LangChain versions ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data/processed/stoic_corpus.txt")
DB_PATH = os.path.join(BASE_DIR, "data/chroma_db")

def build_db():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print(f"🗑️  Deleted old database at {DB_PATH}")

    print(f"--- 1. Loading Stoic Corpus from {DATA_PATH} ---")
    try:
        loader = TextLoader(DATA_PATH, encoding="utf-8")
        documents = loader.load()
    except FileNotFoundError:
        print(f"❌ Error: Could not find {DATA_PATH}")
        print("Did you run 'scripts/02_cleaning_text.py'?")
        return
    
    print("--- 2. Splitting Text into Chunks ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✂️  Created {len(chunks)} chunks.")

    print("--- 3. Creating Embeddings (The Math) ---")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print(f"--- 4. Saving to ChromaDB at {DB_PATH} ---")
    Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model, 
        persist_directory=DB_PATH
    )
    print("✅ Success! Database built.")

if __name__ == "__main__":
    build_db()