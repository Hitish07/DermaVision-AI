from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.database import Base, engine
from api.routers import scan, history
from api.main import router as auth_router

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(scan.router)
app.include_router(history.router)

@app.get("/")
def read_root():
    return {"message": "Skin Scan AI Backend Running"}
