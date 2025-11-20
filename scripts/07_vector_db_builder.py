import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- CONFIG ---
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..")
DATA_PATH = os.path.join(BASE_DIR, "data/processed/stoic_corpus.txt")
DB_PATH = os.path.join(BASE_DIR, "data/chroma_db")

def build_db():
    # 1. Clear old database
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    print("--- 1. Loading Stoic Corpus ---")
    loader = TextLoader(DATA_PATH, encoding="utf-8")
    documents = loader.load()
    
    print("--- 2. Splitting Text into Chunks ---")
    # We split text into 500-character chunks with 50-char overlap
    # This ensures we capture complete thoughts
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("--- 3. Creating Embeddings (Vectors) ---")
    # This model turns text into numbers
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print(f"--- 4. Saving to ChromaDB at {DB_PATH} ---")
    Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model, 
        persist_directory=DB_PATH
    )
    print("--- Success! Vector DB built. ---")

if __name__ == "__main__":
    build_db()