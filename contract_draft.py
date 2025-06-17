from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from typing import Optional
import uuid
import re
import textwrap

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

# Improved formatting function - less spacing, better structure
def smart_text_cleanup(text: str, max_line_length: int = 80) -> str:
    """
    Smart text cleanup with proper spacing and complete content
    """
    if not text:
        return text
    
    # STEP 1: Remove markdown formatting
    patterns_to_remove = [
        r'```[\w\s]*\n',
        r'```[\w\s]*',
        r'\n```\n',
        r'\n```',
        r'```\n',
        r'```',
        r'`{1,3}',
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.MULTILINE)
    
    # Remove markdown formatting
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    
    # Fix escaped quotes
    text = text.replace('\\"', '"')
    text = text.replace("\\'", "'")
    text = text.replace('\\n', '\n')
    
    # STEP 2: Smart spacing - reduce excessive newlines but preserve structure
    # Replace 4+ newlines with 2 newlines (double spacing)
    text = re.sub(r'\n{4,}', '\n\n', text)
    # Replace 3 newlines with 2 newlines
    text = re.sub(r'\n{3}', '\n\n', text)
    
    # STEP 3: Process lines for wrapping
    lines = text.split('\n')
    processed_lines = []
    
    for line in lines:
        # Keep empty lines but don't add extra ones
        if not line.strip():
            # Only add empty line if the last line wasn't empty
            if processed_lines and processed_lines[-1] != '':
                processed_lines.append('')
            continue
        
        cleaned_line = line.strip()
        
        # Don't wrap if line is reasonable length
        if len(cleaned_line) <= max_line_length:
            processed_lines.append(cleaned_line)
            continue
        
        # Handle different content types
        
        # Section headers (ALL CAPS, reasonably short)
        if (cleaned_line.isupper() and 
            len(cleaned_line.split()) <= 8):
            processed_lines.append(cleaned_line)
            continue
        
        # Numbered clauses with better formatting
        numbered_match = re.match(r'^(\d+\.)\s*([A-Z][^:]*):?\s*(.*)', cleaned_line)
        if numbered_match:
            number = numbered_match.group(1)
            title = numbered_match.group(2)
            content = numbered_match.group(3)
            
            # Create header
            if content:
                header = f"{number} {title.upper()}:"
                processed_lines.append(header)
                
                # Wrap content with minimal indent
                if len(content) <= max_line_length - 3:
                    processed_lines.append(f"   {content}")
                else:
                    wrapped_content = textwrap.fill(
                        content,
                        width=max_line_length - 3,
                        initial_indent='   ',
                        subsequent_indent='   ',
                        break_long_words=False,
                        break_on_hyphens=True
                    )
                    processed_lines.append(wrapped_content)
            else:
                processed_lines.append(f"{number} {title.upper()}")
            continue
        
        # Lettered subsections
        letter_match = re.match(r'^([A-Z]\.)\s*(.*)', cleaned_line)
        if letter_match:
            letter = letter_match.group(1)
            content = letter_match.group(2)
            
            if len(f"{letter} {content}") <= max_line_length:
                processed_lines.append(f"{letter} {content}")
            else:
                wrapped = textwrap.fill(
                    f"{letter} {content}",
                    width=max_line_length,
                    subsequent_indent='   ',
                    break_long_words=False,
                    break_on_hyphens=True
                )
                processed_lines.append(wrapped)
            continue
        
        # Regular long lines - wrap normally
        wrapped = textwrap.fill(
            cleaned_line,
            width=max_line_length,
            break_long_words=False,
            break_on_hyphens=True
        )
        processed_lines.append(wrapped)
    
    # STEP 4: Final assembly with controlled spacing
    result = '\n'.join(processed_lines)
    
    # Final cleanup - max 2 consecutive newlines
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    return result.strip()

# Enhanced prompt for complete contracts
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

    IMPORTANT: Generate a COMPLETE, COMPREHENSIVE contract with ALL necessary legal clauses and provisions. Do not provide abbreviated or summary versions.

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

# Main endpoint with improved formatting
@app.post("/legal", response_model=ContractResponse)
async def indian_legal_assistant(request: ContractRequest):
    """
    Endpoint with smart formatting - complete content, proper spacing
    """
    try:
        session_id = get_or_create_session(request.session_id)
        session_data = user_sessions[session_id]
        
        # Create prompt for comprehensive response
        prompt = create_indian_legal_prompt_with_context(
            request.query, 
            session_data["conversation_history"]
        )
        
        # Generate response
        response = model.generate_content(prompt)
        
        if response and response.text:
            # Smart cleanup with proper spacing
            clean_response = smart_text_cleanup(response.text, max_line_length=80)
            
            # Check for rejection
            rejection_message = "I am a contract maker specialized in Indian legal system. I don't have knowledge outside legal and contract matters."
            
            # Store interaction
            interaction = {
                "query": request.query,
                "response": clean_response,
                "timestamp": str(uuid.uuid1().time)
            }
            session_data["conversation_history"].append(interaction)
            session_data["total_queries"] += 1
            
            if rejection_message in clean_response:
                return ContractResponse(
                    success=False,
                    query=request.query,
                    error=clean_response.strip(),
                    session_id=session_id,
                    total_queries=session_data["total_queries"]
                )
            else:
                return ContractResponse(
                    success=True,
                    query=request.query,
                    response=clean_response,
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
        "message": "Indian Contract Assistant - Complete & Well-Formatted",
        "description": "Generates complete contracts with proper spacing",
        "features": {
            "complete_contracts": "Full, comprehensive legal documents",
            "smart_spacing": "Proper line spacing without excessive gaps",
            "80_char_limit": "Optimal line length for readability",
            "comprehensive_clauses": "All necessary legal provisions included"
        },
        "improvements": {
            "content_length": "Full detailed contracts, not summaries",
            "spacing_control": "Reduced excessive line breaks",
            "better_structure": "Organized sections with proper headers"
        },
        "active_sessions": len(user_sessions)
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Indian Contract Assistant",
        "formatting": "complete-and-clean",
        "content": "comprehensive",
        "active_sessions": len(user_sessions)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("contract:app", host="0.0.0.0", port=9000, reload=True)
