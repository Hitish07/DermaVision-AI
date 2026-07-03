from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from datetime import timedelta
from sqlalchemy.orm import Session

from .database import engine, Base, get_db
from .routers import scan, history
from .auth import (
    create_access_token,
    verify_password,
    get_user_by_username,
    create_user,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)

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


class AuthRequest(BaseModel):
    username: str
    password: str


@app.post("/register")
def register(credentials: AuthRequest, db: Session = Depends(get_db)):
    """Register a new user account"""
    if len(credentials.username.strip()) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
    if len(credentials.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    existing_user = get_user_by_username(db, credentials.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already taken")

    create_user(db, credentials.username, credentials.password)

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": credentials.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/login")
def login(credentials: LoginRequest, db: Session = Depends(get_db)):
    """Login endpoint - returns JWT token"""
    user = get_user_by_username(db, credentials.username)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not verify_password(credentials.password, user.hashed_password):
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
