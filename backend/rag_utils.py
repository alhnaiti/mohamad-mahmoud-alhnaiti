# rag_utils backend
import os
import numpy as np
import requests
import pdfplumber
from concurrent.futures import ThreadPoolExecutor
from chromadb import Client
from chromadb.config import Settings
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import tempfile


def load_pdf(file_path):
    pages_text = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text and text.strip():
                pages_text.append(text)
            else:
                pages_text.append("")
    with tempfile.TemporaryDirectory() as temp_dir:
        images = convert_from_path(file_path, output_folder=temp_dir)
        for i, img in enumerate(images):
            if not pages_text[i].strip():  
                ocr_text = pytesseract.image_to_string(img, lang="eng+ara")
                pages_text[i] = ocr_text

    pages_text = [t.replace("\n", " ").strip() for t in pages_text]
    return pages_text

def split_text(docs, chunk_size=1000, overlap=100):
    chunks = []
    for doc in docs:
        start = 0
        while start < len(doc):
            end = start + chunk_size
            chunks.append(doc[start:end])
            start += chunk_size - overlap
    return chunks

def get_embedding(text, model=os.environ.get("EMBEDDING_MODEL")
): # the model is secret 
    ollama_base = os.environ.get("OLLAMA_BASE_URL") #secret
    try:
        response = requests.post(
            f"{ollama_base}/v1/embeddings",
            json={"model":  model,  "input":text}
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    except Exception as e:
        print(f"⚠️ Embedding failed for text chunk: {e}")
        raise RuntimeError("Embedding generation failed.")



def get_chroma_client():
    return Client(Settings(
        anonymized_telemetry=False,
        allow_reset=True,
        persist_directory="chroma_db",
        
        is_persistent=True
    ))

def build_vector_db(chunks, model=os.environ.get("EMBEDDING_MODEL")
): #secret
    client = get_chroma_client()
    
    
    try:
        client.delete_collection("pdf_docs")
    except:
        pass  
    
    collection = client.create_collection(name="pdf_docs")

    def embed(chunk):
        return get_embedding(chunk, model=model)

    with ThreadPoolExecutor(max_workers=3) as executor:
        embeddings = list(executor.map(embed, chunks))

    import uuid
    unique_ids = [str(uuid.uuid4()) for _ in range(len(chunks))]

    collection.add(
        ids=unique_ids,  # Use unique ids 
        documents=chunks,
        metadatas=[{"source": f"chunk_{i}"} for i in range(len(chunks))],
        embeddings=embeddings
    )
    return collection

def retrieve_similar(query, vector_db, top_k=3,model=os.environ.get("EMBEDDING_MODEL")
):#secret
    query_vector = get_embedding(query, model=model)
    results = vector_db.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents"]
    )
    top_chunks = results['documents'][0]
    return "\n\n".join(top_chunks)