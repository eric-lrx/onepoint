from fastapi import APIRouter, HTTPException
import subprocess
from llm_deepseek import ask_ai as llm_ask  # Rename the imported function

router = APIRouter(prefix="/api")

@router.get("/ping", tags=["Health Check"])
async def ping():
    """
    Health check endpoint to verify the API is running.
    """
    return {"message": "pong"}

@router.get("/ask", tags=["Artificial Intelligence"])
async def ask_ai_endpoint(question: str):  # Rename the route function
    """
    Endpoint to ask a question to the AI model.
    """
    politesse_mot_banni = ["vous plaît", "merci", "svp", "bonjour", "salut", "te plait", "cordialement", "sincèrement", "bien à vous", "bien cordialement", "respectueusement", "salutations", "merci d'avance", "merci beaucoup", "aurevoir", "à bientôt", "à plus tard", "à la prochaine", "à tout à l'heure", "à demain", "à tout de suite", "à très vite", "à très bientôt", "à très plus tard", "à très la prochaine", "à très tout à l'heure", "à très demain", "à très tout de suite"]

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if len(question) > 1000:
        raise HTTPException(status_code=400, detail="Question is too long")
    if any(word in question.lower() for word in politesse_mot_banni):
        return {"response": "Pas besoins de formule de politesse, au contraire, vous me faites utiliser de l'énergie précieuse pour rien."}

    # AI model logic would go here
    question = question.replace("%20", " ")
    response = llm_ask(question)  # Use the renamed import
    
    return {"response": response}