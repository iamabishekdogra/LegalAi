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

# AGGRESSIVE formatting function to remove ALL markdown
def clean_and_format_text(text: str, max_line_length: int = 70) -> str:
    """
    Aggressively clean and format text for frontend display
    """
    if not text:
        return text
    
    # STEP 1: AGGRESSIVELY REMOVE ALL MARKDOWN
    # Remove code blocks - multiple patterns
    text = re.sub(r'```[\w\s]*\n', '', text)  # Opening code blocks
    text = re.sub(r'```[\w\s]*', '', text)    # Code blocks without newline
    text = re.sub(r'\n```\n', '\n', text)     # Closing code blocks with newlines
    text = re.sub(r'\n```', '', text)         # Closing code blocks
    text = re.sub(r'```', '', text)           # Any remaining backticks
    
    # Remove all backticks
    text = text.replace('`', '')
    
    # Remove markdown bold/italic
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    
    # Remove markdown headers
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    
    # Clean up quotes that got escaped
    text = text.replace('\\"', '"')
    text = text.replace("\\'", "'")
    
    # STEP 2: NORMALIZE WHITESPACE
    # Replace multiple spaces with single space (except at line start)
    text = re.sub(r'(?<!^) {2,}', ' ', text, flags=re.MULTILINE)
    
    # Clean up excessive newlines but preserve document structure
    text = re.sub(r'\n{4,}', '\n\n\n', text)  # Max 3 newlines
    
    # STEP 3: PROCESS LINE BY LINE FOR WRAPPING
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        # Keep empty lines for document structure
        if not line.strip():
            formatted_lines.append('')
            continue
        
        stripped_line = line.strip()
        
        # Don't wrap short lines
        if len(stripped_line) <= max_line_length:
            formatted_lines.append(stripped_line)
            continue
        
        # Handle different types of content
        
        # Legal section headers (ALL CAPS, short)
        if (stripped_line.isupper() and 
            len(stripped_line.split()) <= 6 and 
            not stripped_line.endswith(':')):
            formatted_lines.append(stripped_line)
            continue
        
        # Numbered clauses (1., 2., etc.)
        if re.match(r'^\d+\.\s+[A-Z]', stripped_line):
            # Split at first colon or after first sentence
            if ':' in stripped_line:
                parts = stripped_line.split(':', 1)
                header = parts[0] + ':'
                content = parts[1].strip()
                
                formatted_lines.append(header)
                if content:
                    wrapped_content = textwrap.fill(
                        content,
                        width=max_line_length - 3,
                        initial_indent='   ',
                        subsequent_indent='   ',
                        break_long_words=False
                    )
                    formatted_lines.append(wrapped_content)
            else:
                # No colon, wrap the whole thing
                wrapped = textwrap.fill(
                    stripped_line,
                    width=max_line_length,
                    subsequent_indent='   ',
                    break_long_words=False
                )
                formatted_lines.append(wrapped)
            continue
        
        # Lettered subsections (A., B., etc.)
        if re.match(r'^[A-Z]\.\s+', stripped_line):
            wrapped = textwrap.fill(
                stripped_line,
                width=max_line_length,
                subsequent_indent='   ',
                break_long_words=False,
                break_on_hyphens=False
            )
            formatted_lines.append(wrapped)
            continue
        
        # Legal party descriptions (long lines with legal language)
        if ('hereinafter referred to as' in stripped_line or 
            'which expression shall' in stripped_line or
            'residing at' in stripped_line):
            # These are legal boilerplate - wrap more carefully
            wrapped = textwrap.fill(
                stripped_line,
                width=max_line_length,
                break_long_words=False,
                break_on_hyphens=False,
                expand_tabs=False
            )
            formatted_lines.append(wrapped)
            continue
        
        # Regular long lines
        wrapped = textwrap.fill(
            stripped_line,
            width=max_line_length,
            break_long_words=False,
            break_on_hyphens=True
        )
        formatted_lines.append(wrapped)
    
    # STEP 4: FINAL CLEANUP
    result = '\n'.join(formatted_lines)
    
    # Remove any remaining excessive whitespace
    result = re.sub(r'\n{4,}', '\n\n\n', result)
    
    # Clean up trailing spaces on lines
    result = '\n'.join(line.rstrip() for line in result.split('\n'))
    
    # Remove leading/trailing whitespace from entire text
    result = result.strip()
    
    return result

# Updated prompt with even stricter instructions
def create_indian_legal_prompt_with_context(query: str, conversation_history: list) -> str:
    context = ""
    if conversation_history:
        context = "\n\nPREVIOUS CONVERSATION CONTEXT:\n"
        for i, item in enumerate(conversation_history[-3:], 1):
            context += f"{i}. Previous Query: {item['query'][:100]}...\n"
            context += f"   Previous Response: {item['response'][:200]}...\n"
        context += "\nUse this context to provide relevant and consistent responses.\n"
    
    return f"""
    You are an expert Indian contract attorney specialized in Indian legal system.

ABSOLUTE FORMATTING REQUIREMENTS - NO EXCEPTIONS:
1. NEVER use markdown code blocks (``` or ```)
2. NEVER use backticks of any kind
3. NEVER use asterisks for bold (**)
4. Make sure to use proepr Subheadings and Titles
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

    CONTENT RULES:
    If the query is NOT about Indian legal/contract matters, respond exactly:
    "I am a contract maker specialized in Indian legal system. I don't have knowledge outside legal and contract matters. Please ask questions about contracts, agreements, or Indian legal framework."

    If the query IS about contracts:
    - Draft complete contracts following Indian Contract Act 1872
    - Use proper legal terminology
    - Include all necessary clauses
    - Reference Indian jurisdiction

    EXAMPLE OF CORRECT FORMAT (no code blocks):

    LEASE AGREEMENT

    THIS LEASE AGREEMENT is made and entered into at [City], India on this [Date].

    BETWEEN

    [Name of Lessor], residing at [Address], hereinafter referred to as the "Lessor" of the ONE PART;

    AND

    [Name of Lessee], residing at [Address], hereinafter referred to as the "Lessee" of the OTHER PART.

    1. LEASE TERM: The Lessor hereby leases the property to the Lessee for a period of [Duration].

    2. RENT: The monthly rent shall be Rs. [Amount] payable in advance.

    {context}

    User Query: {query}

    CRITICAL: Start your response immediately with the document title. Do not use any formatting symbols or code blocks.

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

# Main endpoint with enhanced formatting
@app.post("/legal", response_model=ContractResponse)
async def indian_legal_assistant(request: ContractRequest):
    """
    Enhanced endpoint with aggressive formatting cleanup
    """
    try:
        session_id = get_or_create_session(request.session_id)
        session_data = user_sessions[session_id]
        
        # Create prompt
        prompt = create_indian_legal_prompt_with_context(
            request.query, 
            session_data["conversation_history"]
        )
        
        # Generate response
        response = model.generate_content(prompt)
        
        if response and response.text:
            # AGGRESSIVELY clean the response
            cleaned_response = clean_and_format_text(response.text, max_line_length=70)
            
            # Check for rejection
            rejection_message = "I am a contract maker specialized in Indian legal system. I don't have knowledge outside legal and contract matters."
            
            # Store interaction
            interaction = {
                "query": request.query,
                "response": cleaned_response,
                "timestamp": str(uuid.uuid1().time)
            }
            session_data["conversation_history"].append(interaction)
            session_data["total_queries"] += 1
            
            if rejection_message in cleaned_response:
                return ContractResponse(
                    success=False,
                    query=request.query,
                    error=cleaned_response.strip(),
                    session_id=session_id,
                    total_queries=session_data["total_queries"]
                )
            else:
                return ContractResponse(
                    success=True,
                    query=request.query,
                    response=cleaned_response,
                    session_id=session_id,
                    total_queries=session_data["total_queries"]
                )
        else:
            raise HTTPException(status_code=500, detail="Failed to generate response.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Other endpoints remain the same
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
        "message": "Indian Contract Assistant - Zero Formatting",
        "description": "Aggressively removes ALL markdown formatting",
        "features": {
            "no_code_blocks": "Removes all ``` blocks",
            "no_markdown": "Removes **, *, #, etc.",
            "line_wrapping": "70 character limit",
            "plain_text_only": "Pure text output"
        },
        "endpoints": {
            "main": "/legal (POST)",
            "history": "/session/{session_id} (GET)",
            "refresh": "/refresh (POST)"
        },
        "active_sessions": len(user_sessions)
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Indian Contract Assistant",
        "formatting": "zero-markdown",
        "active_sessions": len(user_sessions)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("contract_draft:app", host="0.0.0.0", port=9000, reload=True)
