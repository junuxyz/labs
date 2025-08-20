import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model import generate_response
from pydantic import BaseModel

app = FastAPI(title="simple chat")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    max_length: int = 100


class ChatResponse(BaseModel):
    response: str


@app.get("/")
async def root():
    return {"message": "simple chat is running."}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        model_response = generate_response(request.message, request.max_length)
        return ChatResponse(response=model_response)
    except Exception as e:
        return ChatResponse(response=f"Error: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
