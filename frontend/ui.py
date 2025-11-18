import streamlit as st
import requests
import os

# --- CONFIGURATION ---
# When running in Docker, both services are in the same container.
# But Streamlit runs on port 8501, FastAPI on 7860.
# We need to tell Streamlit where the API is.
API_URL = os.getenv("API_URL", "http://localhost:7860")

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Sokratik",
    page_icon="🏛️",
    layout="centered"
)

# --- HEADER ---
st.title("🏛️ SOKRATIK")
st.caption("Speak with the wisdom of the Stoics.")

# --- SESSION STATE ---
# We need to remember the chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Greetings. I am ready to discuss life, virtue, and the nature of things. What is on your mind?"}
    ]

# --- DISPLAY CHAT HISTORY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- USER INPUT ---
if prompt := st.chat_input("Ask your question..."):
    # 1. Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Get response from API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            # Call the FastAPI backend
            response = requests.post(
                f"{API_URL}/generate", 
                json={"prompt": prompt},
                timeout=120 # Give the model time to think
            )
            
            if response.status_code == 200:
                answer = response.json()["answer"]
                message_placeholder.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                error_msg = f"Error: {response.status_code} - {response.text}"
                message_placeholder.error(error_msg)
        
        except Exception as e:
            message_placeholder.error(f"Connection Error: {e}")