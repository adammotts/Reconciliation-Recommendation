# main.py
from fastapi import FastAPI
from app.api.endpoints import reconciliation, health
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings

app = FastAPI()

# CORS configuration
origins = [
    "http://localhost",
    "http://192.168.1.34:8000",
    "http://localhost:8000",
    "http://localhost:3000",  # Add your frontend URL here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the users router

app.include_router(reconciliation.router, prefix="/reconciliation", tags=["reconciliations"])

app.include_router(health.router, prefix="", tags=["Health"])