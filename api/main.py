from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .database import engine, Base
from .routers import scan, history

# Create DB tables if they don't exist
Base.metadata.create_all(bind=engine)

app = FastAPI(title="DermaVision AI Clinical Platform", version="4.0.0")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Directories for images/artifacts
app.mount("/figures", StaticFiles(directory="figures"), name="figures")
app.mount("/uploads", StaticFiles(directory="api/uploads"), name="uploads")

# Include Routers
app.include_router(scan.router)
app.include_router(history.router)
