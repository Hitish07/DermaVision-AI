from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from datetime import timedelta

from .database import engine, Base
from .routers import scan, history
from .auth import create_access_token, verify_password, USERS_DB, ACCESS_TOKEN_EXPIRE_MINUTES

# Create DB tables if they don't exist
Base.metadata.create_all(bind=engine)

app = FastAPI(title="DermaVision AI Clinical Platform", version="4.0.0")

# CORS Middleware - Fixed for Production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://derma-vision-ai-dun.vercel.app",
        "https://hitish1220-dermavision-ai-backend.hf.space",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ROOT ENDPOINTS
@app.get("/")
def read_root():
    return {
        "message": "DermaVision AI Clinical Platform is running!",
        "version": "4.0.0",
        "status": "ok"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/login")
def login(credentials: LoginRequest):
    """Login endpoint - returns JWT token"""
    if credentials.username not in USERS_DB:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not verify_password(credentials.password, USERS_DB[credentials.username]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": credentials.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Mount Static Directories for images/artifacts
app.mount("/figures", StaticFiles(directory="figures"), name="figures")
app.mount("/uploads", StaticFiles(directory="api/uploads"), name="uploads")

# Include Routers
app.include_router(scan.router)
app.include_router(history.router)