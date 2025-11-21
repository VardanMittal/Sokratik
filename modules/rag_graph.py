from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..")
DB_PATH=os.path.join(BASE_DIR, "data/chroma_db")

class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str


def retrieve_node(state: AgentState):
    """
    The Librarian: Searches the ChromaDB database for relevant text.
    """
    question = state["question"]
    print(f"--- Graph: Retrieving info for: {question} ---")
    
    # Initialize Embedding Model (Same one we used to build the DB)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Connect to the DB
    db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    
    # Search for top 3 matches
    docs = db.similarity_search(question, k=3)
    context = [d.page_content for d in docs]
    
    return {"context": context}


def generate_node(state: AgentState, llm_engine):
    """
    The Philosopher: Generates the answer using the retrieved context.
    """
    question = state["question"]
    context = state["context"]
    
    # Combine the retrieved quotes into one block of text
    context_text = "\n\n".join(context)
    
    # Create the "Augmented" System Prompt
    system_prompt = (
        "You are Sokratik, a Stoic philosopher. "
        "Answer the user's question using the provided context from Stoic texts. "
        "Do not explicitly mention 'the context' or 'the documents'. "
        "Speak with wisdom and tranquility."
    )
    
    # Format the prompt for Llama 3
    formatted_prompt = (
        f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\nContext:\n{context_text}\n\nQuestion: {question}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    print("--- Graph: Generating answer... ---")
    # Call the GGUF model
    output = llm_engine(
        formatted_prompt,
        max_tokens=256,
        stop=["<|eot_id|>", "<|end_of_text|>"],
        echo=False,
        temperature=0.6
    )
    
    answer = output["choices"][0]["text"].strip()
    return {"answer": answer}

# --- 3. BUILD THE GRAPH ---
def build_graph(llm_instance):
    """
    Compiles the nodes into a runnable application.
    """
    workflow = StateGraph(AgentState)

    # Add the nodes
    workflow.add_node("retrieve", retrieve_node)
    # We use a lambda function to pass the loaded LLM into the node
    workflow.add_node("generate", lambda state: generate_node(state, llm_instance))

    # Define the flow (The Edges)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()