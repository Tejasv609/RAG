from fastapi import FastAPI, File, UploadFile
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

from llama_index.core.node_parser import SentenceSplitter
import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

app = FastAPI()


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load LLaMA model
system_prompt = "You are a Q&A assistant. Your goal is to answer questions based on the uploaded document."
query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name='meta-llama/Llama-2-7b-chat-hf',
    model_name='meta-llama/Llama-2-7b-chat-hf',
    device_map="cpu",  # Force CPU usage
    model_kwargs={"torch_dtype": torch.float32}
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")

settings = {
    "llm": llm,
    "embed_model": embed_model,
    "node_parser": SentenceSplitter(chunk_size=1024)
}

index = None

from fastapi import HTTPException

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handles file upload and indexing"""
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        global index
        documents = SimpleDirectoryReader(UPLOAD_DIR).load_data()
        index = VectorStoreIndex.from_documents(documents=documents, service_context=settings)

        return {"message": f"File '{file.filename}' uploaded and indexed successfully."}

    except Exception as e:
        print(f"Error: {e}")  # Print error in terminal
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/ask")
async def ask_question(question: str):
    """Handles querying of indexed documents"""
    if index is None:
        return {"error": "No document uploaded."}
    
    query_engine = index.as_query_engine()
    response = query_engine.query(question)

    return {"answer": str(response)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
