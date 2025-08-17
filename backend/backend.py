import os
import logging
import json
import shutil # Import the shutil module for directory deletion
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PyPDF2 import PdfReader

# LangChain components
from langchain_community.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
import os                                                                                                                                                                                                          
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
load_dotenv(Path("../.env"))

# --- Configuration and Initialization ---
UPLOAD_DIR = './uploads'
QDRANT_DATA_PATH = './qdrant_stores'
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(QDRANT_DATA_PATH, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class Question(BaseModel):
    question: str

# --- Global Objects ---
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

llm = ChatGroq(temperature=0, model_name='llama3-70b-8192', api_key=os.getenv("GROQ_API_KEY"))

chat_sessions_cache = {}

# --- Helper Functions ---
def get_chat_path(chat_id: str):
    return os.path.join(QDRANT_DATA_PATH, chat_id)

def get_chat_history_file(chat_path: str):
    return os.path.join(chat_path, "chat_history.json")

def get_metadata_file(chat_path: str):
    return os.path.join(chat_path, "metadata.json")

def load_or_create_chat_session(chat_id: str):
    if chat_id in chat_sessions_cache:
        return chat_sessions_cache[chat_id]

    chat_path = get_chat_path(chat_id)
    os.makedirs(chat_path, exist_ok=True)
    
    vectorstore = None
    if os.path.exists(os.path.join(chat_path, "collection")):
        try:
            vectorstore = Qdrant.from_existing_collection(
                embedding=embeddings, path=chat_path, collection_name=f"chat_{chat_id}"
            )
        except Exception as e:
            logger.error(f"Failed to load vector store for {chat_id}: {e}")

    history_file = get_chat_history_file(chat_path)
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        chat_memory=FileChatMessageHistory(history_file),
        return_messages=True
    )
    
    session = {"vectorstore": vectorstore, "memory": memory}
    chat_sessions_cache[chat_id] = session
    return session

# --- API Endpoints ---
@app.get("/chats")
async def get_all_chats():
    chat_ids = [d for d in os.listdir(QDRANT_DATA_PATH) if os.path.isdir(os.path.join(QDRANT_DATA_PATH, d))]
    chats = {}
    for chat_id in chat_ids:
        metadata_file = get_metadata_file(get_chat_path(chat_id))
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                chats[chat_id] = metadata.get("title", f"Chat {chat_id[:8]}")
    return chats

@app.get("/chat/{chat_id}")
async def get_chat_details(chat_id: str):
    chat_path = get_chat_path(chat_id)
    if not os.path.exists(chat_path):
        raise HTTPException(status_code=404, detail="Chat not found")

    history_file = get_chat_history_file(chat_path)
    history = []
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            data = json.load(f)
            for i in range(0, len(data), 2):
                if i+1 < len(data) and data[i]['type'] == 'human' and data[i+1]['type'] == 'ai':
                    history.append((data[i]['data']['content'], data[i+1]['data']['content']))

    metadata_file = get_metadata_file(chat_path)
    metadata = {}
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
    return {
        "history": history,
        "processed_files": metadata.get("processed_files", []),
        "title": metadata.get("title", f"Chat {chat_id[:8]}")
    }

# --- NEW DELETE ENDPOINT ---
@app.delete("/chat/{chat_id}")
async def delete_chat(chat_id: str):
    """Deletes all data associated with a chat session."""
    if chat_id in chat_sessions_cache:
        del chat_sessions_cache[chat_id]

    chat_path = get_chat_path(chat_id)
    if not os.path.exists(chat_path):
        raise HTTPException(status_code=404, detail="Chat not found")
    
    try:
        shutil.rmtree(chat_path)
        logger.info(f"Successfully deleted chat data for chat_id: {chat_id}")
        return {"message": f"Chat {chat_id} deleted successfully."}
    except Exception as e:
        logger.error(f"Error deleting chat {chat_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete chat data: {e}")


@app.post("/upload/{chat_id}")
async def upload_file(chat_id: str, files: list[UploadFile] = File(...)):
    session = load_or_create_chat_session(chat_id)
    chat_path = get_chat_path(chat_id)
    
    all_text = ""
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        with open(file_path, "rb") as f_rb:
            pdf_reader = PdfReader(f_rb)
            for page in pdf_reader.pages:
                all_text += page.extract_text() or ""
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = [Document(page_content=chunk) for chunk in text_splitter.split_text(text=all_text)]

    if not documents: raise HTTPException(status_code=400, detail="No text extracted.")

    if session["vectorstore"] is None:
        session["vectorstore"] = Qdrant.from_documents(
            documents, embeddings, path=chat_path, collection_name=f"chat_{chat_id}"
        )
    else:
        session["vectorstore"].add_documents(documents)

    metadata_file = get_metadata_file(chat_path)
    new_filenames = sorted([file.filename for file in files])
    
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {"processed_files": [], "title": f"Chat {chat_id[:8]}"}
    
    existing_files = set(metadata.get("processed_files", []))
    for fname in new_filenames:
        existing_files.add(fname)
    
    metadata["processed_files"] = sorted(list(existing_files))
    
    if metadata["title"] == f"Chat {chat_id[:8]}" and new_filenames:
        metadata["title"] = f"Chat with {new_filenames[0]}"

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)
        
    return {"message": "Files processed"}

@app.post("/ask/{chat_id}")
async def ask_question(chat_id: str, question: Question):
    session = load_or_create_chat_session(chat_id)
    if session.get("vectorstore") is None:
        raise HTTPException(status_code=400, detail="No documents for this chat.")
    
    retriever = session["vectorstore"].as_retriever()
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=session["memory"]
    )
    response = chain.invoke({"question": question.question})
    return {"answer": response.get("answer")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
