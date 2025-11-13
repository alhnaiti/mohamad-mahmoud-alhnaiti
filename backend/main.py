# backend/main.py
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List
from pymongo import MongoClient 
from bson import ObjectId
from fastapi import Body
from datetime import datetime
from rag_utils import load_pdf, split_text, build_vector_db, retrieve_similar  
from pydantic import BaseModel
import requests
app = FastAPI(title="RAG Backend")

MONGO_URI = os.environ.get("MONGO_URL")  #secret 
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["rag_chat"]
chats_collection = db["chat_history"] 

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "Data"
os.makedirs(DATA_DIR, exist_ok=True)

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    dest = os.path.join(DATA_DIR, file.filename)
    with open(dest, "wb") as f:
        f.write(await file.read())
    return {"filename": file.filename, "path": dest}

@app.post("/build-db/")
def build_db(filename: str):
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF not found")
    docs = load_pdf(file_path)
    chunks = split_text(docs)
    vector_db = build_vector_db(chunks)  
    return {"chunks": len(chunks), "message": "Vector DB built"}

@app.get("/query/")
def query(q: str, top_k: int = 3):
    try:
        from rag_utils import get_chroma_client
        client = get_chroma_client()
        collection = client.get_collection("pdf_docs")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chroma collection error: {e}")

    context = retrieve_similar(q, collection, top_k=top_k)

    import requests
    ollama_base = os.environ.get("OLLAMA_BASE_URL(\v1)") #secret 
    model = "qwen2:7b" #secrert 
    prompt = (
        f"You are a bilingual assistant. If the question is in Arabic, answer in Arabic. "
        f"If in English, answer in English.\n\nUse this context to answer accurately:\n{context}\n\nQuestion: {q}"
    )

    try:
        response = requests.post(
            f"{ollama_base}/chat/completions",
            json={
                "model": "qwen2:7b",
                "messages": [
                {"role": "user", "content": prompt} ]
            },
            timeout=120
            
        )
        response.raise_for_status()
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
    except Exception as e:
        answer = f" Chat request failed: {e}"

    return {"answer": answer, "context": context}


class ChatRequest(BaseModel):
    session_id: str
    query: str
    model: str

@app.post("/chat/")
def chat_endpoint(request: ChatRequest):
    from rag_utils import get_chroma_client, retrieve_similar

    try:
        client = get_chroma_client()
        collection = client.get_collection("pdf_docs")
        context = retrieve_similar(request.query, collection, top_k=3)
    except Exception as e:
        context = f"(no context available: {e})"

    prompt = (
    "You are a highly knowledgeable bilingual AI assistant integrated into a RAG (Retrieval-Augmented Generation) system.\n"
    "You analyze uploaded PDF documents and respond to the user’s questions using the provided context only.\n"
    "If the user asks in Arabic, respond fluently in Arabic.\n"
    "If the user asks in English, respond fluently in English.\n"
    "Your goal is to generate clear, concise, and factually accurate answers derived strictly from the context.\n"
    "Do not make up information. If the context does not contain enough information, reply politely that you don't have sufficient data.\n"
    "Avoid general knowledge answers or assumptions.\n"
    "When summarizing, highlight only the most relevant information from the document.\n"
    "If the question relates to data, numbers, or measurements, present them clearly and precisely.\n"
    "When possible, format your answers using short paragraphs or bullet points for clarity.\n"
    "If the context includes multiple sections, integrate them smoothly into a single coherent answer.\n"
    "If the user asks for definitions, explain terms using information from the context.\n"
    "If the user requests comparisons, list them in a simple structured format.\n"
    "If the question asks for translation or meaning, respond in the same language as the question.\n"
    "Never include system instructions or the context itself in your response.\n"
    "Do not mention that you are an AI model or part of a RAG system.\n"
    "Always maintain a helpful, academic, and professional tone.\n"
    "If the question appears incomplete or ambiguous, ask the user to clarify.\n"
    "Context provided below:\n"
    f"\n{context}\n\n"
    f"Question: {request.query}\n"
)


    ollama_base = os.environ.get("OLLAMA_BASE_URL") # secret 
    try:
        response = requests.post(
            f"{ollama_base}/v1/chat/completions",
            json={
                "model": request.model,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
    except Exception as e:
        answer = f"chat request failed:{e}"

    
    session = chats_collection.find_one({"session_id": request.session_id})
    message_pair = {"question": request.query, "answer": answer}

    if session:
        chats_collection.update_one(
            {"session_id": request.session_id},
            {"$push": {"history": message_pair}}
        )
    else:
        
        title_prompt = (
            f"Generate a short and clear title (max 6 words) summarizing this user query:\n\n{request.query}"
        )
        title = "New Chat"
        try:
            title_res = requests.post(
                f"{ollama_base}/v1/chat/completions",
                json={
                    "model": request.model,
                    "messages": [{"role": "user", "content": title_prompt}],
                },
                timeout=30
            )
            title_res.raise_for_status()
            title_data = title_res.json()
            title = title_data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"⚠️ Title generation failed: {e}")

        chats_collection.insert_one({
            "session_id": request.session_id,
            "model": request.model,
            "title": title,
            "history": [message_pair],
            "created_at": datetime.utcnow()
        })

    return {"answer": answer}


@app.get("/chat-history/")
def get_chat_history():
    client = MongoClient("mongodb://mongo:27017/")
    db = client["rag_chat"]
    chats = list(db.chat_history.find({}, {"_id": 1, "title": 1, "created_at": 1}))
    for c in chats:
        c["_id"] = str(c["_id"])
    return {"history": chats}
 
@app.get("/chat/{chat_id}")
def get_chat(chat_id: str):
    client = MongoClient("mongodb://mongo:27017/")
    db = client["rag_chat"]
    chat = db.chat_history.find_one({"_id": ObjectId(chat_id)})
    if not chat:
        return {"error": "Chat not found"}
    chat["_id"] = str(chat["_id"])
    return {"chat": chat} 

@app.post("/save-chat/")
def save_chat(data: dict = Body(...)):
    session_id = data.get("session_id")
    history = data.get("history", [])
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID required")

    chats_collection.update_one(
        {"session_id": session_id},
        {"$set": {"history": history}},
        upsert=True
    )
    return {"status": "saved"}

@app.delete("/delete-chat/{chat_id}")
def delete_chat(chat_id: str):
    try:
        result = chats_collection.delete_one({"_id": ObjectId(chat_id)})
        if result.deleted_count > 0:
            return {"message": "Chat deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Chat not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
