import os
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import tempfile
from pathlib import Path
import shutil

# Import our modules
from reading_functions import DocumentReader, TextChunker
from embedding_functions import OllamaEmbeddingClient
from indexing_functions import VectorStore, DocumentChunk
from chatbot_functions import Chatbot, GenerationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG API",
    description="REST API for RAG (Retrieval-Augmented Generation) using local models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3:instruct"
PERSIST_DIR = "./vector_store"

# Create instances
embedding_client = OllamaEmbeddingClient()
vector_store = VectorStore(persist_dir=PERSIST_DIR, dimension=768)  # nomic-embed-text uses 768 dimensions
chatbot = Chatbot(model_name=LLM_MODEL)

# Request/Response Models
class DocumentUpload(BaseModel):
    file_path: str
    metadata: Optional[Dict[str, Any]] = {}

class DocumentResponse(BaseModel):
    document_id: str
    chunk_count: int
    metadata: Dict[str, Any]

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    filter_conditions: Optional[Dict[str, Any]] = None
    generation_config: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

class ChatRequest(BaseModel):
    message: str
    chat_history: Optional[List[Dict[str, str]]] = None
    top_k: int = 5
    filter_conditions: Optional[Dict[str, Any]] = None
    generation_config: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]

# Utility Functions
def get_generation_config(config_dict: Optional[Dict[str, Any]] = None) -> GenerationConfig:
    """Create a GenerationConfig from a dictionary."""
    if config_dict is None:
        return GenerationConfig()
    
    # Only include valid fields for GenerationConfig
    valid_fields = {f for f in GenerationConfig.__annotations__}
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
    return GenerationConfig(**filtered_config)

# API Endpoints
@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    metadata: str = Form("{}")
):
    """
    Upload a document to the vector store.
    
    The document will be processed, chunked, and embedded.
    """
    try:
        # Parse metadata
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            metadata_dict = {}
        
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            # Read the document
            reader = DocumentReader()
            text, doc_type = reader.read_document(tmp_path)
            
            # Add to vector store
            document_id = vector_store.add_document(
                text=text,
                metadata={
                    **metadata_dict,
                    'filename': file.filename,
                    'content_type': file.content_type,
                    'doc_type': doc_type
                },
                embeddings_client=embedding_client
            )
            
            # Get document info
            doc_info = vector_store.documents.get(document_id, {})
            
            return {
                'document_id': document_id,
                'chunk_count': doc_info.get('chunk_count', 0),
                'metadata': doc_info.get('metadata', {})
            }
            
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Error deleting temp file: {e}")
    
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.get("/documents", response_model=List[Dict[str, Any]])
async def list_documents():
    """List all documents in the vector store."""
    return [
        {
            'document_id': doc_id,
            'metadata': doc_info.get('metadata', {}),
            'chunk_count': doc_info.get('chunk_count', 0)
        }
        for doc_id, doc_info in vector_store.documents.items()
    ]

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all its chunks from the vector store."""
    if document_id not in vector_store.documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    vector_store.delete_document(document_id)
    return {"status": "success", "message": f"Document {document_id} deleted"}

@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest
):
    """
    Query the document store and generate an answer.
    
    This performs retrieval-augmented generation using the provided question.
    """
    try:
        # Search for relevant chunks
        search_results = vector_store.search(
            query=request.question,
            k=request.top_k,
            filter_conditions=request.filter_conditions,
            embeddings_client=embedding_client
        )
        
        # Extract chunks and format context
        chunks = [result[0] for result in search_results]
        context = [chunk.text for chunk in chunks]
        
        # Get generation config
        gen_config = get_generation_config(request.generation_config)
        
        # Generate answer using the chatbot
        answer = chatbot.chat(
            message=request.question,
            context=context,
            config=gen_config
        )
        
        # Format sources
        sources = [
            {
                'text': chunk.text,
                'metadata': chunk.metadata,
                'score': float(score)
            }
            for chunk, score in search_results
        ]
        
        return {
            'answer': answer,
            'sources': sources
        }
    
    except Exception as e:
        logger.error(f"Error in query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest
):
    """
    Chat with the assistant using the document store for context.
    
    This maintains conversation history and uses it for context.
    """
    try:
        # If chat history is provided, create a new chatbot instance with it
        current_chatbot = chatbot
        if request.chat_history:
            current_chatbot = Chatbot(model_name=LLM_MODEL)
            for msg in request.chat_history:
                if msg['role'] == 'user':
                    current_chatbot.chat(msg['content'])
                elif msg['role'] == 'assistant':
                    current_chatbot.conversation_history.append({
                        'role': 'assistant',
                        'content': msg['content']
                    })
        
        # Search for relevant chunks
        search_results = vector_store.search(
            query=request.message,
            k=request.top_k,
            filter_conditions=request.filter_conditions,
            embeddings_client=embedding_client
        )
        
        # Extract chunks and format context
        context = [chunk.text for chunk, _ in search_results]
        
        # Get generation config
        gen_config = get_generation_config(request.generation_config)
        
        # Generate response
        response = current_chatbot.chat(
            message=request.message,
            context=context,
            config=gen_config
        )
        
        # Format sources
        sources = [
            {
                'text': chunk.text,
                'metadata': chunk.metadata,
                'score': float(score)
            }
            for chunk, score in search_results
        ]
        
        return {
            'response': response,
            'sources': sources
        }
    
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
