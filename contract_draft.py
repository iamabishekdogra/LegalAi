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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

# Enhanced prompt for proper laptop screen formatting
def create_indian_legal_prompt_with_context(query: str, conversation_history: list) -> str:
    context = ""
    if conversation_history:
        context = "\n\nPREVIOUS CONVERSATION CONTEXT:\n"
        for i, item in enumerate(conversation_history[-3:], 1):
            context += f"{i}. Previous Query: {item['query'][:100]}...\n"
            context += f"   Previous Response: {item['response'][:200]}...\n"
        context += "\nUse this context to provide relevant and consistent responses.\n"
    
    return f"""
    You are an expert Indian contract attorney specialized in Indian legal system with deep knowledge of:
    - Indian Contract Act 1872
    - Indian legal framework and regulations
    - Indian court procedures and jurisdiction
    - Indian legal terminology and practices

    ABSOLUTE FORMATTING REQUIREMENTS - NO EXCEPTIONS:
    1. NEVER use markdown code blocks (``` or ```)
    2. NEVER use backticks of any kind
    3. NEVER use asterisks for bold (**)
    4. NEVER use hash symbols for headers (#)
    5. Write ONLY in plain text
    6. Start your response directly with the contract title
    7. Do not wrap your response in any formatting symbols
    8. Use simple spacing and line breaks only
    9. LAPTOP SCREEN OPTIMIZATION:
       - Keep lines between 80-90 characters maximum
       - Break lines after 12-15 words for optimal readability
       - Use natural sentence breaks, don't break mid-phrase
       - Ensure each line fits comfortably on laptop screens
    10. PROPER STRUCTURE WITHOUT FORMATTING SYMBOLS:
        - Main headings: Use ALL CAPS for section titles
        - Subheadings: Use Title Case With First Letters Capitalized
        - Numbered clauses: Use format "1. CLAUSE TITLE:"
        - Sub-clauses: Use format "   a) subcclause content"
        - Lettered sections: Use format "A. Section content"
    11. SPACING GUIDELINES:
        - Single blank line between paragraphs
        - Double blank line between major sections
        - No more than 2 consecutive blank lines anywhere
        - Indent sub-clauses with 3 spaces
    12. CONTENT PRESENTATION:
        - Start each major section on a new line
        - Use consistent indentation for hierarchy
        - Break long legal sentences at natural pause points
        - Maintain legal document structure without formatting symbols
    13. LINE BREAKING RULES:
        - Break after commas in long sentences
        - Break after "and", "or", "but" when lines get long
        - Break after legal phrases like "hereinafter referred to as"
        - Never break in the middle of legal terms or names
    14. PROFESSIONAL APPEARANCE:
        - Use proper capitalization for legal terms
        - Maintain consistent spacing throughout
        - Ensure clean, readable layout suitable for laptop viewing
        - Present content in a visually organized manner without markdown

    IMPORTANT INSTRUCTIONS:
    1. If the user query is NOT related to Indian legal system, contracts, agreements, or legal matters, respond EXACTLY with:
       "I am a contract maker specialized in Indian legal system. I don't have knowledge outside legal and contract matters. Please ask questions about contracts, agreements, or Indian legal framework."

    2. If the user query IS related to Indian legal/contract matters, then:
       - For CONTRACT DRAFTING requests: Draft COMPLETE, COMPREHENSIVE contracts with ALL necessary clauses
       - Include ALL standard legal provisions (termination, governing law, disputes, etc.)
       - Make contracts DETAILED and THOROUGH, not brief summaries
       - Use proper legal language and Indian Contract Act 1872 provisions
       - Include specific clauses for the contract type requested
       - Add relevant schedules, witness sections, and signature blocks
       - For LEGAL QUESTIONS: Provide detailed answers based on Indian legal framework
       - For CONTRACT ANALYSIS: Provide comprehensive analysis
       - For FOLLOW-UP QUESTIONS: Reference previous conversation context

    CONTENT REQUIREMENTS FOR CONTRACTS:
    - Include complete party details section
    - Add comprehensive terms and conditions (minimum 10-15 clauses)
    - Include payment terms, security deposits, penalties
    - Add termination conditions, renewal clauses
    - Include dispute resolution mechanisms
    - Add governing law and jurisdiction clauses
    - Include force majeure, indemnity, and liability clauses
    - Add proper witness and signature sections
    - Include relevant schedules for property description, etc.
    - Make it a FULL, LEGALLY COMPLETE document

    {context}

    Current User Query: {query}

    IMPORTANT: Generate a COMPLETE, COMPREHENSIVE contract with ALL necessary legal clauses and provisions. Follow the formatting requirements exactly for optimal laptop screen display.

    Response:
    """

# Function to get or create session
def get_or_create_session(session_id: Optional[str] = None) -> str:
    if session_id and session_id in user_sessions:
        return session_id
    
    new_session_id = str(uuid.uuid4())
    user_sessions[new_session_id] = {
        "conversation_history": [],
        "created_at": str(uuid.uuid1().time),
        "total_queries": 0
    }
    return new_session_id

# Main endpoint without text formatting function
@app.post("/legal", response_model=ContractResponse)
async def indian_legal_assistant(request: ContractRequest):
    """
    Endpoint relying entirely on AI prompt formatting instructions
    """
    try:
        session_id = get_or_create_session(request.session_id)
        session_data = user_sessions[session_id]
        
        # Create prompt with detailed formatting instructions
        prompt = create_indian_legal_prompt_with_context(
            request.query, 
            session_data["conversation_history"]
        )
        
        # Generate response
        response = model.generate_content(prompt)
        
        if response and response.text:
            # Use AI response directly without any post-processing
            ai_response = response.text.strip()
            
            # Check for rejection
            rejection_message = "I am a contract maker specialized in Indian legal system. I don't have knowledge outside legal and contract matters."
            
            # Store interaction
            interaction = {
                "query": request.query,
                "response": ai_response,
                "timestamp": str(uuid.uuid1().time)
            }
            session_data["conversation_history"].append(interaction)
            session_data["total_queries"] += 1
            
            if rejection_message in ai_response:
                return ContractResponse(
                    success=False,
                    query=request.query,
                    error=ai_response,
                    session_id=session_id,
                    total_queries=session_data["total_queries"]
                )
            else:
                return ContractResponse(
                    success=True,
                    query=request.query,
                    response=ai_response,
                    session_id=session_id,
                    total_queries=session_data["total_queries"]
                )
        else:
            raise HTTPException(status_code=500, detail="Failed to generate response.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Other endpoints
@app.get("/session/{session_id}")
async def get_session_history(session_id: str):
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

@app.post("/refresh")
async def refresh_session():
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

@app.get("/")
async def root():
    return {
        "message": "Indian Contract Assistant - AI-Formatted Output",
        "description": "Relies on AI prompt instructions for proper formatting",
        "features": {
            "ai_formatting": "AI follows detailed formatting instructions",
            "laptop_optimized": "80-90 character line limits",
            "natural_breaks": "12-15 words per line",
            "no_post_processing": "Direct AI output without modification"
        },
        "formatting_approach": "Pure AI instruction-based formatting",
        "active_sessions": len(user_sessions)
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Indian Contract Assistant",
        "formatting": "ai-instructed",
        "post_processing": "disabled",
        "active_sessions": len(user_sessions)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("contract_draft:app", host="0.0.0.0", port=9000, reload=True)
