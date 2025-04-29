from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api")

@router.get("/ping", tags=["Health Check"])
async def ping():
    """
    Health check endpoint to verify the API is running.
    """
    return {"message": "pong"}

@router.post("/ask", tags=["Artificial Intelligence"])
async def ask_ai(question: str):
    """
    Endpoint to ask a question to the AI model.
    """
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # AI model logic would go here
    response = ""
    
    return {"response": response}

