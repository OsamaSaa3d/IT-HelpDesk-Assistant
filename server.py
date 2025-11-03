from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from dotenv import load_dotenv
import uvicorn

from src.faiss_index import FAISSIndex
from src.embeddings import EmbeddingModel
from src.llm_client import TicketResolutionAssistant
from src.config import GEMINI_API_KEY, DEFAULT_TOP_K

load_dotenv()

app = FastAPI(
    title="Ticket Support System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

faiss_index = None
embedding_model = None
ai_assistant = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20)


class AIRecommendationRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20)


class TicketResult(BaseModel):
    id: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    results: List[TicketResult]
    total: int


class AIRecommendationResponse(BaseModel):
    query: str
    recommendation: str
    candidates: List[Dict[str, Any]]
    backend: str


@app.on_event("startup")
async def startup_event():
    global faiss_index, embedding_model, ai_assistant
    
    print("Loading FAISS index and embedding model...")
    faiss_index = FAISSIndex()
    faiss_index.load()
    
    embedding_model = EmbeddingModel()
    
    if GEMINI_API_KEY:
        print("Initializing AI assistant...")
        ai_assistant = TicketResolutionAssistant()
    else:
        print("Warning: GEMINI_API_KEY not set. AI recommendations disabled.")
    
    print("Server ready!")


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    if not faiss_index or not embedding_model:
        raise HTTPException(status_code=503, detail="Search index not loaded")
    
    try:
        results = faiss_index.query(
            request.query,
            embedding_model=embedding_model,
            top_k=request.top_k
        )
        
        return SearchResponse(
            query=request.query,
            results=[TicketResult(**r) for r in results],
            total=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/recommend", response_model=AIRecommendationResponse)
async def get_recommendation(request: AIRecommendationRequest):
    if not ai_assistant:
        raise HTTPException(
            status_code=503,
            detail="AI assistant not available. Please set GEMINI_API_KEY."
        )
    
    if not faiss_index or not embedding_model:
        raise HTTPException(status_code=503, detail="Search index not loaded")
    
    try:
        similar_tickets = faiss_index.query(
            request.query,
            embedding_model=embedding_model,
            top_k=request.top_k
        )
        
        recommendation = ai_assistant.generate_recommendation(
            request.query,
            similar_tickets
        )
        
        return AIRecommendationResponse(
            query=request.query,
            recommendation=recommendation["llm_output"],
            candidates=recommendation["candidates"],
            backend=recommendation["backend"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

