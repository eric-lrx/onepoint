from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import sys
import os

# Add parent directory to path to import llm_deepseek
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_deepseek import Call_Ollama, classify_prompt

router = APIRouter(prefix="/api")

@router.get("/ping", tags=["Health Check"])
async def ping():
    """
    Health check endpoint to verify the API is running.
    """
    return {"message": "pong"}

class QuestionRequest(BaseModel):
    question: str

@router.post("/ask", tags=["Artificial Intelligence"])
async def ask_ai(question: str = Query(...)):
    """
    Endpoint to ask a question to the AI model.
    """
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Classify the prompt to determine if we should call Ollama
        should_call, reason = classify_prompt(question)
        
        if should_call:
            # Call the Ollama model with the question
            response = Call_Ollama(question)
        else:
            response = "Désolé, je ne peux pas traiter cette demande. " + reason
        
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")