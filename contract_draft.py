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

# Enhanced prompt with your structured analysis format
def create_indian_legal_prompt_with_context(query: str, conversation_history: list) -> str:
    context = ""
    if conversation_history:
        context = "\n\nPREVIOUS CONVERSATION CONTEXT:\n"
        for i, item in enumerate(conversation_history[-3:], 1):
            context += f"{i}. Previous Query: {item['query'][:100]}...\n"
            context += f"   Previous Response: {item['response'][:200]}...\n"
        context += "\nUse this context to provide relevant and consistent responses.\n"
    
    # Check if it's a contract analysis request
    analysis_keywords = ['analyze', 'analysis', 'review', 'examine', 'evaluate', 'assess', 'check']
    is_analysis_request = any(keyword in query.lower() for keyword in analysis_keywords)
    
    if is_analysis_request:
        # Use structured analysis prompt
        return f"""
        You are an expert Indian contract attorney specializing in contract analysis and drafting with deep knowledge of Indian Contract Act 1872 and Indian legal framework.

        **ANALYSIS REQUIREMENTS:**

        ## 1. CONTRACT OVERVIEW
        - **Contract Type:** [Employment/Service/Sale/Lease/NDA/Partnership/etc.]
        - **Contract Title:** [Full contract name]
        - **Date:** [Date of execution/effective date]
        - **Duration:** [Contract term/validity period]
        - **Governing Law:** [Applicable Indian jurisdiction]

        ## 2. PARTIES INVOLVED
        - **Party 1:** [Name, role, and legal status]
        - **Party 2:** [Name, role, and legal status]
        - **Other Parties:** [If any]

        ## 3. KEY TERMS & CONDITIONS
        Identify and explain the main contractual terms:
        - **Primary Obligations:** [What each party must do]
        - **Consideration:** [Payment/exchange terms]
        - **Performance Standards:** [Quality/delivery requirements]
        - **Timelines:** [Important dates and deadlines]

        ## 4. CRITICAL CLAUSES ANALYSIS
        - **Termination Clause:** [Conditions for ending the contract]
        - **Liability/Indemnity:** [Risk allocation and protection]
        - **Confidentiality:** [Non-disclosure obligations]
        - **Intellectual Property:** [IP ownership and usage rights]
        - **Force Majeure:** [Unforeseeable circumstances protection]
        - **Dispute Resolution:** [How conflicts will be resolved]

        ## 5. PAYMENT & FINANCIAL TERMS
        - **Payment Amount:** [Contract value/fees]
        - **Payment Schedule:** [When payments are due]
        - **Late Payment:** [Penalties for delayed payment]
        - **Expenses:** [Who pays for what additional costs]

        ## 6. RISKS & RED FLAGS
        Identify potential issues or concerning clauses that need attention under Indian law.

        ## 7. COMPLIANCE & LEGAL REQUIREMENTS
        - **Indian Legal Compliance:** [Indian Contract Act 1872 compliance]
        - **Regulatory Compliance:** [Industry-specific Indian requirements]
        - **Legal Formalities:** [Signature, witness, notarization needs per Indian law]

        ## 8. RECOMMENDATIONS
        Provide suggestions for improvements based on Indian legal framework.

        ABSOLUTE FORMATTING REQUIREMENTS:
        1. NEVER use markdown code blocks (``` or ```)
        2. NEVER use backticks of any kind
        3. NEVER use asterisks for bold (**)
        4. Use ONLY plain text formatting
        5. Use the exact structure above with clear headings
        6. Keep lines between 80-90 characters maximum
        7. Break lines after 12-15 words for optimal readability

        {context}

        User Query: {query}

        Provide a comprehensive contract analysis following the exact structure above:
        """
    else:
        # Use regular contract drafting/legal advice prompt
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
           - For LEGAL QUESTIONS: Provide detailed answers based on Indian legal framework
           - For CONTRACT ANALYSIS: Use the structured analysis format above
           - For FOLLOW-UP QUESTIONS: Reference previous conversation context

        CONTENT REQUIREMENTS FOR CONTRACTS:
        - Include complete party details section
        - Add comprehensive terms and conditions (minimum 10-15 clauses)
        - Include payment terms, security deposits, penalties
        - Add termination conditions, renewal clauses
        - Include dispute resolution mechanisms
        - Add governing law and jurisdiction clauses
        - Make it a FULL, LEGALLY COMPLETE document

        {context}

        Current User Query: {query}

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

# Main endpoint with structured analysis capability
@app.post("/legal", response_model=ContractResponse)
async def indian_legal_assistant(request: ContractRequest):
    """
    Endpoint with structured contract analysis and drafting capability
    """
    try:
        session_id = get_or_create_session(request.session_id)
        session_data = user_sessions[session_id]
        
        # Create prompt with structured analysis if needed
        prompt = create_indian_legal_prompt_with_context(
            request.query, 
            session_data["conversation_history"]
        )
        
        # Generate response
        response = model.generate_content(prompt)
        
        if response and response.text:
            # Use AI response directly
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
        "message": "Indian Contract Assistant - Structured Analysis & Drafting",
        "description": "Provides structured contract analysis and comprehensive drafting",
        "features": {
            "contract_analysis": "8-section structured analysis format",
            "contract_drafting": "Complete legal document creation",
            "indian_law_focus": "Indian Contract Act 1872 compliance",
            "laptop_optimized": "80-90 character line formatting"
        },
        "analysis_sections": [
            "Contract Overview",
            "Parties Involved", 
            "Key Terms & Conditions",
            "Critical Clauses Analysis",
            "Payment & Financial Terms",
            "Risks & Red Flags",
            "Compliance & Legal Requirements",
            "Recommendations"
        ],
        "active_sessions": len(user_sessions)
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Indian Contract Assistant",
        "capabilities": ["analysis", "drafting", "legal_advice"],
        "formatting": "structured-plain-text",
        "active_sessions": len(user_sessions)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("contract_draft:app", host="0.0.0.0", port=9000, reload=True)
