from fastapi import FastAPI
from models import Query
from rag_engine import RAGEngine

app = FastAPI(title="XYZ Real Estate RAG Service")
rag = RAGEngine()

@app.post("/ask")
def ask_question(query: Query):
    result = rag.ask_with_sources(query.question, session_id=query.session_id)
    return {
        "question": query.question, 
        "answer": result["answer"],
        "sources": result["sources"]
    }