import streamlit as st
import requests

# --- CONFIGURATION ---
# REPLACE THIS with your actual Backend Space URL
# It usually looks like: https://vardan10-sokratik-api.hf.space
BACKEND_URL = "https://vardan10-sokratik-api.hf.space" 

st.set_page_config(page_title="Sokratik", page_icon="🏛️")

# --- HEADER ---
st.title("🏛️ SOKRATIK")
st.caption(f"RAG Agent • Connected to Brain at: {BACKEND_URL}")

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "I am ready. Ask me about the Stoics."}
    ]

# --- CHAT HISTORY ---
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- USER INPUT ---
if prompt := st.chat_input("What is on your mind?"):
    # 1. Show User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 2. Get Response from Backend
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.write("Thinking & Reading...")
        
        try:
            # Send request to your FastAPI Backend
            resp = requests.post(
                f"{BACKEND_URL}/generate", 
                json={"prompt": prompt},
                timeout=120 # Give the backend time to think
            )
            
            if resp.status_code == 200:
                data = resp.json()
                answer = data["answer"]
                meta = data.get("meta", {})
                
                # Show the answer
                placeholder.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Show "Thinking Process" (Optional, looks cool!)
                with st.expander("See Thinking Process"):
                    st.write(f"⏱️ Duration: {meta.get('duration', 0)}s")
                    st.write(f"📚 Documents Read: {meta.get('retrieved_docs', 0)}")
                    
            else:
                placeholder.error(f"Error {resp.status_code}: {resp.text}")
                
        except Exception as e:
            placeholder.error(f"Connection Failed. Is the backend running? Error: {e}")