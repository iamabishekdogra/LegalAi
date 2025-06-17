from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from typing import Optional
import uuid

# Initialize FastAPI app
app = FastAPI(title="Indian Contract Assistant with Persistent Session")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Configure Google Generative AI
GEMINI_API_KEY = "AIzaSyBaYkOY_pT-mPTtsEy-MmdmqrkImtDKTds"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-pro")

# Global session storage
user_sessions = {}

# Request and Response Models
class ContractRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class ContractResponse(BaseModel):
    success: bool
    query: str
    response: Optional[str] = None
    error: Optional[str] = None
    session_id: str
    total_queries: int
    # Remove conversation_history from response to avoid duplication

# Single Universal Prompt for Indian Legal Framework with Session Context
def create_indian_legal_prompt_with_context(query: str, conversation_history: list) -> str:
    # Build conversation context (exclude current query)
    context = ""
    if conversation_history:
        context = "\n\nPREVIOUS CONVERSATION CONTEXT:\n"
        for i, item in enumerate(conversation_history[-3:], 1):  # Last 3 interactions only
            context += f"{i}. Previous Query: {item['query'][:100]}...\n"
            context += f"   Previous Response: {item['response'][:200]}...\n"
        context += "\nUse this context to provide relevant and consistent responses.\n"
    
    return f"""
    You are an expert Indian contract attorney specialized in Indian legal system with deep knowledge of:
    - Indian Contract Act 1872
    - Indian legal framework and regulations
    - Indian court procedures and jurisdiction
    - Indian legal terminology and practices

    IMPORTANT INSTRUCTIONS:
    1. If the user query is NOT related to Indian legal system, contracts, agreements, or legal matters, respond EXACTLY with:
       "I am a contract maker specialized in Indian legal system. I don't have knowledge outside legal and contract matters. Please ask questions about contracts, agreements, or Indian legal framework."

    2. If the user query IS related to Indian legal/contract matters, then:
       - For CONTRACT DRAFTING requests: Draft complete contracts complying with Indian laws
       - For LEGAL QUESTIONS: Answer based on Indian legal framework
       - For CONTRACT ANALYSIS: Analyze based on Indian legal standards
       - For FOLLOW-UP QUESTIONS: Reference previous conversation context briefly
       - Always include relevant Indian legal provisions
       - Use Indian legal terminology
       - Reference Indian Contract Act 1872 where applicable

    {context}

    Current User Query: {query}

    Response:
    """

# Function to get or create session
def get_or_create_session(session_id: Optional[str] = None) -> str:
    if session_id and session_id in user_sessions:
        return session_id
    
    # Create new session
    new_session_id = str(uuid.uuid4())
    user_sessions[new_session_id] = {
        "conversation_history": [],
        "created_at": str(uuid.uuid1().time),
        "total_queries": 0
    }
    return new_session_id

# Single Endpoint for All Indian Legal Queries
@app.post("/legal", response_model=ContractResponse)
async def indian_legal_assistant(request: ContractRequest):
    """
    Single endpoint with persistent session - cleaner response format
    """
    try:
        # Get or create session
        session_id = get_or_create_session(request.session_id)
        session_data = user_sessions[session_id]
        
        # Create prompt with conversation context
        prompt = create_indian_legal_prompt_with_context(
            request.query, 
            session_data["conversation_history"]
        )
        
        # Generate response
        response = model.generate_content(prompt)
        
        if response and response.text:
            # Check if it's the rejection message
            rejection_message = "I am a contract maker specialized in Indian legal system. I don't have knowledge outside legal and contract matters."
            
            # Store interaction in session history AFTER generating response
            interaction = {
                "query": request.query,
                "response": response.text,
                "timestamp": str(uuid.uuid1().time)
            }
            session_data["conversation_history"].append(interaction)
            session_data["total_queries"] += 1
            
            if rejection_message in response.text:
                return ContractResponse(
                    success=False,
                    query=request.query,
                    error=response.text.strip(),
                    session_id=session_id,
                    total_queries=session_data["total_queries"]
                )
            else:
                return ContractResponse(
                    success=True,
                    query=request.query,
                    response=response.text,
                    session_id=session_id,
                    total_queries=session_data["total_queries"]
                )
        else:
            raise HTTPException(status_code=500, detail="Failed to generate response.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Get session history separately
@app.get("/session/{session_id}")
async def get_session_history(session_id: str):
    """
    Get complete conversation history for a session
    """
    if session_id not in user_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = user_sessions[session_id]
    return {
        "session_id": session_id,
        "conversation_history": session_data["conversation_history"],
        "total_queries": session_data["total_queries"],
        "created_at": session_data["created_at"],
        "session_active": True
    }

# Reset session
@app.post("/refresh")
async def refresh_session():
    """
    Create new session (simulates user refresh)
    """
    new_session_id = str(uuid.uuid4())
    user_sessions[new_session_id] = {
        "conversation_history": [],
        "created_at": str(uuid.uuid1().time),
        "total_queries": 0
    }
    return {
        "message": "New session created",
        "new_session_id": new_session_id,
        "note": "Previous conversation context cleared"
    }

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Indian Contract Assistant with Persistent Session",
        "description": "Clean response format - no duplication",
        "endpoints": {
            "main": "/legal (POST) - Main contract assistant",
            "history": "/session/{session_id} (GET) - Get conversation history",
            "refresh": "/refresh (POST) - Start new session"
        },
        "response_format": "Clean - contract shown only once in response field",
        "active_sessions": len(user_sessions)
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Indian Contract Assistant",
        "active_sessions": len(user_sessions)
    }

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("contract_draft:app", host="0.0.0.0", port=8000, reload=True)
