import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

# Import our modules
from document_processor import DocumentProcessor
from embedding_service import EmbeddingService
from vector_store import VectorStore
from qa_service import QAService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize services
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

# Initialize services
embedding_service = EmbeddingService()
vector_store = VectorStore()
qa_service = QAService()
document_processor = DocumentProcessor()

# Initialize FastAPI app
app = FastAPI(
    title="Research Assistant API",
    description="API for document-based question answering and challenge generation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class DocumentUploadResponse(BaseModel):
    document_id: str
    summary: str
    num_chunks: int

class QuestionRequest(BaseModel):
    document_id: str
    question: str

class QuestionResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

class ChallengeRequest(BaseModel):
    document_id: str
    num_questions: int = 3

class ChallengeQuestion(BaseModel):
    question: str
    answer: str

class ChallengeResponse(BaseModel):
    questions: List[ChallengeQuestion]

class AnswerRequest(BaseModel):
    document_id: str
    question: str
    user_answer: str
    reference_answer: str

class AnswerResponse(BaseModel):
    score: float
    feedback: str

# Helper Functions
def get_document_chunks(document_id: str) -> List[Dict[str, Any]]:
    """Get all chunks for a document."""
    # This is a simplified version - in a real app, you'd query the vector store
    return []

# API Endpoints
@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...)
):
    """Upload and process a document."""
    try:
        # Save uploaded file
        file_ext = Path(file.filename).suffix
        file_id = str(uuid.uuid4())
        file_path = UPLOAD_DIR / f"{file_id}{file_ext}"
        
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        
        # Process the document
        try:
            # Read and chunk the document
            text, doc_type = document_processor.read_document(str(file_path))
            chunks = document_processor.chunk_text(text)
            
            # Generate embeddings for chunks
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = embedding_service.get_embeddings_batch(chunk_texts)
            
            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk['embedding'] = embedding
            
            # Generate a summary
            summary = document_processor.generate_summary(text)
            
            # Store in vector database
            vector_store.add_document(
                doc_id=file_id,
                chunks=chunks,
                metadata={
                    'filename': file.filename,
                    'content_type': file.content_type,
                    'type': doc_type,
                    'summary': summary
                }
            )
            
            return {
                'document_id': file_id,
                'summary': summary,
                'num_chunks': len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
            
        finally:
            # Clean up the uploaded file
            try:
                file_path.unlink()
            except Exception as e:
                logger.warning(f"Error deleting temporary file: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about a document."""
    try:
        # Get the most relevant chunks
        query_embedding = embedding_service.get_embedding(request.question)
        results = vector_store.search(
            query_embedding=query_embedding,
            doc_id=request.document_id,
            k=3
        )
        
        # Extract context
        context = [result['text'] for result in results]
        
        # Generate answer
        response = qa_service.answer_question(
            question=request.question,
            context=context
        )
        
        return {
            'answer': response['answer'],
            'sources': [
                {
                    'text': text,
                    'similarity': result['score']
                }
                for text, result in zip(context, results)
            ]
        }
        
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")

@app.post("/challenge", response_model=ChallengeResponse)
async def generate_challenge(request: ChallengeRequest):
    """Generate challenge questions for a document."""
    try:
        # Get random chunks from the document
        chunks = get_document_chunks(request.document_id)
        if not chunks:
            raise HTTPException(status_code=404, detail="Document not found or has no chunks")
        
        # Use the first few chunks as context
        context = [chunk['text'] for chunk in chunks[:5]]
        
        # Generate questions
        questions = qa_service.generate_challenge_questions(
            context=context,
            num_questions=request.num_questions
        )
        
        return {
            'questions': [
                {
                    'question': q['question'],
                    'answer': q['answer']
                }
                for q in questions
            ]
        }
        
    except Exception as e:
        logger.error(f"Error generating challenge: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating challenge: {str(e)}")

@app.post("/evaluate", response_model=AnswerResponse)
async def evaluate_answer(request: AnswerRequest):
    """Evaluate a user's answer to a question."""
    try:
        evaluation = qa_service.evaluate_answer(
            question=request.question,
            user_answer=request.user_answer,
            reference_answer=request.reference_answer
        )
        
        return evaluation
        
    except Exception as e:
        logger.error(f"Error evaluating answer: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error evaluating answer: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
