import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import router as api_router

app = FastAPI(title="EcoGPT API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.include_router(api_router)

if __name__ == "__main__":
    # Run the application using Uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)