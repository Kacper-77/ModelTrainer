from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import train, auth
from .database import Base, engine

app = FastAPI()

Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(train.router)
app.include_router(auth.router)


@app.get("/")
async def root():
    return {"message": "Hello from Model Trainer!"}
