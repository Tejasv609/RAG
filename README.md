# RAG Document Q&A Application

A full-stack Retrieval-Augmented Generation (RAG) application that allows users to upload documents and query them locally using an LLM. This application uses Fastapi, React, and LlamaIndex.

## Features

- **Document Upload**: Upload text or PDF files.
- **Local Indexing**: Generates embeddings locally using `sentence-transformers`.
- **Query Engine**: Uses LLaMA-2 to generate answers based on the uploaded document.
- **FastAPI Backend**: Efficient API endpoints for uploading and querying.
- **React Frontend**: Clean and responsive user interface for document question-answering.

## Technology Stack

- **Backend**: FastAPI, LlamaIndex, HuggingFace (`Llama-2-7b-chat-hf`), PyTorch
- **Frontend**: React, Axios
- **Embeddings**: `sentence-transformers/all-mpnet-base-v2`

## Getting Started

### Prerequisites
- Node.js
- Python 3.8+
- HuggingFace API Token (If leveraging specific gated models like LLaMA-2)

### Backend Setup

1. **Navigate to the backend directory:**
   ```sh
   cd document-qa/backend
   ```
2. **Create and activate a virtual environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Environment Variables:**
   Create a `.env` file in the backend folder and add your HuggingFace token:
   ```env
   HF_TOKEN=your_huggingface_api_token
   ```
5. **Run the FastAPI server:**
   ```sh
   python app.py
   ```
   The backend should now be running at `http://localhost:8000`.

### Frontend Setup

1. **Navigate to the frontend directory:**
   ```sh
   cd frontend
   ```
2. **Install dependencies:**
   ```sh
   npm install
   ```
3. **Run the React application:**
   ```sh
   npm start
   ```
   The frontend should now be running at `http://localhost:3000`.

## Usage

1. Open the frontend URL in your browser.
2. Click **Choose File** to select your document.
3. Click **Upload** to index the file.
4. Once indexed, type your question in the text box and click **Ask** to receive an AI-generated answer.

## License
MIT
