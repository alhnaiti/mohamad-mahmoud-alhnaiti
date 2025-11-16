# RAG_demo.py  frontend 
import os
import streamlit as st
import requests
import uuid

BACKEND_URL = os.environ.get("BACKEND_URL") 
st.set_page_config(page_title="RAG Chat", layout="wide")

st.title("")

with st.sidebar:
    st.header(" Upload a PDF file")
    uploaded_file = st.file_uploader("", type=["pdf"])
    

    if uploaded_file is not None:
        files = {"file": (uploaded_file.name, uploaded_file.getbuffer(), "application/pdf")}
        res = requests.post(f"{BACKEND_URL}/upload-pdf/", files=files)
        if res.status_code == 200:
            filename = res.json()["filename"]
            st.success(f"File `{filename}` uploaded successfully! ")
            build_res = requests.post(f"{BACKEND_URL}/build-db/", params={"filename": filename})
            if build_res.status_code == 200:
                st.success(f" PDF processed — {build_res.json().get('chunks')} chunks ready.")
            else:
                st.error(f" Build DB error: {build_res.text}")
        else:
            st.error(f" Upload error: {res.text}")
    else:
        st.info("Please upload a PDF file to get started.")
    
    st.subheader("Select Chat Model")
    model_choice = st.selectbox(
        "Choose a model to generate answers:",
        ["qwen2:7b", "phi3", "llama3"],
        index=0
    )
    
    if st.button(" Clear Chat History"):
        st.session_state.history = []
    if st.button("New Chat"):
    
       if "history" in st.session_state and len(st.session_state.history) > 0:
         requests.post(
            f"{BACKEND_URL}/save-chat/",
            json={
                "session_id": st.session_state.session_id,
                "history": st.session_state.history
            }
        )
         st.success("chat saved successfully.")
       else:
          st.info("No chat messages to save, skipping storage.")   
       st.session_state.history = []
       st.session_state.session_id = str(uuid.uuid4())
       st.session_state.show_previous = True 

    if st.session_state.get("show_previous", False):
       st.subheader("Previous Chats")

       try:
        response = requests.get(f"{BACKEND_URL}/chat-history/")
        if response.status_code == 200:
            history = response.json().get("history", [])

            if len(history) == 0:
                st.info("No saved chats found.")
            else:
                for chat in history:
                    chat_id = chat["_id"]
                    chat_title = chat.get("title", f"Chat {chat_id[:6]}")

                    col1, col2 = st.columns([4, 1])
                    with col1:
                        if st.button(chat_title, key=f"load_{chat_id}"):
                            res_chat = requests.get(f"{BACKEND_URL}/chat/{chat_id}")
                            if res_chat.status_code == 200:
                                st.session_state.history = res_chat.json()["chat"]["history"]
                                st.session_state.current_chat_id = chat_id
                                st.success(f"Loaded chat: {chat_title}")
                                st.rerun()

                    with col2:
                        if st.button("❌ ", key=f"delete_{chat_id}"):
                            del_res = requests.delete(f"{BACKEND_URL}/delete-chat/{chat_id}")
                            if del_res.status_code == 200:
                                st.success("Chat deleted ")
                                st.rerun()
                            else:
                                st.error("Failed to delete ")
        else:
            st.error(f"Failed to load chat history: {response.text}")
       except Exception as e:
        st.error(f"Error loading chat history: {e}")        

st.header(" Chat with your document")

if "history" not in st.session_state:
    st.session_state.history = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

for chat_item in st.session_state.history:
    with st.chat_message(name="user"):
        st.markdown(chat_item["question"])
    with st.chat_message(name="assistant"):
        st.markdown(chat_item["answer"])

user_query = st.chat_input("Ask your question in Arabic or English...")

if user_query:
    with st.spinner(f"Retrieving and generating answer using ***{model_choice}**...."):
        response = requests.post(
            f"{BACKEND_URL}/chat/",
            json={"session_id": st.session_state.session_id, "query": user_query, "model": model_choice}
        )

        if response.status_code == 200:
            answer = response.json().get("answer", "No response received.")
        else:
            answer = f"Error: {response.text}"

    st.session_state.history.append({"question": user_query, "answer": answer})
    st.rerun()
