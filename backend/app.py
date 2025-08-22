from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(
    title="AI Chat App",
    description="AI Chat Application with Token Tracking and Cost Monitoring",
    version="1.0.0"
)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

class ChatMessage(BaseModel):
    message: str
    model: str = "gemini-1.5-flash"

class ChatResponse(BaseModel):
    response: str
    tokens_used: int
    cost: float

def calculate_cost(input_tokens: int, output_tokens: int, model: str = "gemini-1.5-flash") -> float:
    """Calculate cost based on Gemini pricing"""
    # Gemini 1.5 Flash pricing (per 1M tokens)
    if model == "gemini-1.5-flash":
        input_cost_per_1m = 0.075  # $0.075 per 1M input tokens
        output_cost_per_1m = 0.30  # $0.30 per 1M output tokens
    else:
        input_cost_per_1m = 0.075
        output_cost_per_1m = 0.30
    
    input_cost = (input_tokens / 1_000_000) * input_cost_per_1m
    output_cost = (output_tokens / 1_000_000) * output_cost_per_1m
    
    return input_cost + output_cost

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main HTML page"""
    with open("../frontend/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "AI Chat App is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_message: ChatMessage):
    """Main chat endpoint with Gemini integration"""
    try:
        # Initialize the model
        model = genai.GenerativeModel(chat_message.model)
        
        # Generate response
        response = model.generate_content(chat_message.message)
        
        # Get token usage (approximate - Gemini doesn't provide exact counts)
        input_tokens = len(chat_message.message.split()) * 1.3  # Rough estimate
        output_tokens = len(response.text.split()) * 1.3
        
        # Calculate cost
        cost = calculate_cost(int(input_tokens), int(output_tokens), chat_message.model)
        
        return ChatResponse(
            response=response.text,
            tokens_used=int(input_tokens + output_tokens),
            cost=round(cost, 6)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)