from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from main import ChatbotManager

app = FastAPI()

# More permissive CORS settings for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Set to False when using allow_origins=["*"]
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

chatbot = ChatbotManager()

class UserMessage(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(user_msg: UserMessage):
    try:
        print(f"📩 Received message: '{user_msg.message}'")
        response_text = chatbot.get_response(user_msg.message)
        print(f"📤 Sending response: '{response_text}'")
        return {"response": response_text}
    except Exception as e:
        print(f"❌ Error processing message: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Lab of Future Chatbot API is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "chatbot": "ready", "timestamp": "2024"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Lab of Future Chatbot API...")
    print("📡 Server will be available at: http://localhost:8000")
    print("🔧 Health check endpoint: http://localhost:8000/health")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)