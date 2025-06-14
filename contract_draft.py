import os
import fitz  # PyMuPDF
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from pathlib import Path
import re
import uuid

# ========== In-memory storage for contract sessions ========== #
contract_sessions = {}

# ========== Unified Pydantic Models ONLY ========== #
class UnifiedContractRequest(BaseModel):
    query: str  # Natural language query - model will understand the intent
    session_id: Optional[str] = None  # For operations on existing contracts

class UnifiedContractResponse(BaseModel):
    success: bool
    query: str
    session_id: Optional[str] = None
    contract_text: Optional[str] = None
    contract_type: Optional[str] = None
    answer: Optional[str] = None
    is_relevant: Optional[bool] = None
    contract_analysis: Optional[str] = None
    key_clauses: Optional[List[str]] = None
    contract_details: Optional[dict] = None
    modification_history: Optional[List[dict]] = None
    detected_intent: Optional[str] = None  # What the model understood user wants
    error: Optional[str] = None

# Add new prompt for intent detection
# Add new prompt for intent detection
QUERY_INTENT_DETECTION_PROMPT = """
You are an expert contract attorney AI. Analyze the user's query and determine what they want to do.

**USER QUERY:** {query}
**SESSION EXISTS:** {has_session}

**POSSIBLE INTENTS:**

1. **DRAFT** - User wants to create/draft a new contract
   - Keywords: "draft", "create", "make", "write", "generate", "new contract"
   - Examples: "Draft a lease agreement", "Create employment contract", "Make a service agreement"
   - Must explicitly ask to CREATE or DRAFT something new

2. **QUESTION** - User wants to ask about an existing contract (requires session)
   - Keywords: "what is", "what are", "how much", "when", "who", "explain", "tell me about", "missing", "included", "contains"
   - Examples: "What is the rent?", "What are payment terms?", "Who are the parties?", "What are missing clauses?"
   - Questions ABOUT existing contract content

3. **MODIFY** - User wants to change/update an existing contract (requires session)
   - Keywords: "add", "remove", "change", "modify", "update", "edit", "include", "insert", "delete"
   - Examples: "Add a pet clause", "Change the rent to $2000", "Remove termination notice"
   - Must be clear instruction to CHANGE something

4. **ANALYZE** - User wants analysis of an existing contract (requires session)
   - Keywords: "analyze", "review", "summary", "risks", "analysis", "breakdown", "evaluate"
   - Examples: "Analyze this contract", "What are the risks?", "Give me a summary", "Review the agreement"
   - Requests for comprehensive analysis or review

**IMPORTANT RULES:**
- If query asks "what are missing" or "what is missing" → QUESTION (not DRAFT)
- If query asks about existing contract content → QUESTION
- If query explicitly says "draft", "create", "make" → DRAFT
- If query says "add", "change", "modify" → MODIFY
- If query says "analyze", "review", "summary" → ANALYZE
- If no session exists, only DRAFT is possible
- If session exists, determine based on query content carefully

**INSTRUCTIONS:**
- Respond with ONLY the intent name: "DRAFT", "QUESTION", "MODIFY", or "ANALYZE"
- If unclear or irrelevant, respond with "INVALID"
- Be very careful with questions vs drafting requests

Intent:
"""
# ========== FastAPI App ========== #
app = FastAPI(
    title="Unified Contract Drafting API", 
    description="Single endpoint for all contract operations: draft, modify, question, analyze"
)

# ========== Add CORS Middleware ========== #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Gemini AI Setup ========== #
GEMINI_API_KEY = "AIzaSyBaYkOY_pT-mPTtsEy-MmdmqrkImtDKTds"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-pro")

# ========== Prompt Templates ========== #
CONTRACT_ANALYSIS_PROMPT = """
You are an expert contract attorney specializing in contract analysis and drafting. Analyze the following contract document and provide a comprehensive, well-structured analysis.

**ANALYSIS REQUIREMENTS:**

## 1. CONTRACT OVERVIEW
- **Contract Type:** [Employment/Service/Sale/Lease/NDA/Partnership/etc.]
- **Contract Title:** [Full contract name]
- **Date:** [Date of execution/effective date]
- **Duration:** [Contract term/validity period]
- **Governing Law:** [Applicable jurisdiction]

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
Identify potential issues or concerning clauses that need attention.

## 7. COMPLIANCE & LEGAL REQUIREMENTS
- **Regulatory Compliance:** [Industry-specific requirements]
- **Legal Formalities:** [Signature, witness, notarization needs]

## 8. RECOMMENDATIONS
Provide suggestions for improvements or modifications.

---
**CONTRACT DOCUMENT:**
{contract_text}

**Instructions:** 
- Be precise and legally accurate
- Use clear headings and bullet points
- If any information is not available, state "Not specified in the contract"
- Focus on contractual obligations and legal implications
- Highlight any unusual or potentially problematic clauses
"""

CONTRACT_RELEVANCE_CHECK_PROMPT = """
You are a strict contract law specialist. Your job is to determine if a user's request is specifically about contract drafting, contract analysis, or contract-related legal work.

**ONLY ACCEPT THESE CONTRACT-RELATED REQUESTS:**
- Draft/create/write a contract (any type)
- Analyze/review an existing contract
- Modify/edit/update a contract
- Ask questions about contract terms, clauses, or legal provisions
- Contract law advice and guidance
- Specific contract types (employment, lease, purchase, service, etc.)

**REJECT ALL OTHER REQUESTS INCLUDING:**
- General greetings (hi, hello, how are you)
- Personal questions
- Technology questions
- General knowledge questions
- Non-legal business questions
- Any topic not directly related to contracts

**USER REQUEST:** {question}

**INSTRUCTIONS:**
Respond with ONLY "RELEVANT" if the request is specifically about contracts, contract drafting, or contract law.
Respond with ONLY "IRRELEVANT" if the request is about anything else, including greetings or general conversation.

Be extremely strict - only genuine contract-related requests should be marked as RELEVANT.
"""

CONTRACT_QA_PROMPT = """
You are a senior contract attorney with extensive experience in contract drafting and analysis. A client has asked you a specific question about a contract. Provide a comprehensive, professional response that directly addresses their question.

**CONTRACT DOCUMENT:**
{contract_text}

**CLIENT QUESTION:**
{question}

**INSTRUCTIONS FOR YOUR RESPONSE:**

1. **Stay Contract-Focused:** Only answer questions related to contracts, contract law, and legal agreements
2. **Professional Tone:** Respond as a legal professional would to a client
3. **Be Comprehensive:** Include all relevant contract details that relate to the question
4. **Legal Accuracy:** Use precise legal terminology and be legally accurate
5. **Reference Contract:** Quote specific clauses or sections when relevant
6. **Provide Context:** Give necessary background about contractual implications
7. **Risk Assessment:** Highlight any legal risks or considerations
8. **Actionable Advice:** Provide practical guidance when appropriate

**RESPONSE GUIDELINES:**
- For clause interpretation: Explain the meaning, implications, and potential risks
- For obligation questions: Clarify what each party must do under the contract
- For modification questions: Explain the process and legal requirements
- For breach questions: Outline consequences and available remedies
- For compliance questions: Detail regulatory requirements and best practices

**Important:** Base your answer strictly on the provided contract document and general contract law principles. If specific information is not in the contract, state "This specific detail is not addressed in the contract document."

Respond professionally as if advising a client on their contractual matter.
"""

SIMPLE_CONTRACT_DRAFTING_PROMPT = """
You are an expert contract attorney. A client has requested a contract with the following query:

**CLIENT REQUEST:** {query}

**INSTRUCTIONS:**
1. Analyze the request to determine the contract type and key requirements
2. Extract any parties, terms, or specific requirements mentioned
3. Draft a complete, professional contract based on the request
4. If specific details are missing, use reasonable defaults for that contract type
5. Include all standard legal clauses appropriate for the contract type

**CONTRACT REQUIREMENTS:**
- Use proper legal formatting with numbered sections
- Include clear headings and subheadings
- Use precise legal language
- Include all essential clauses (parties, consideration, terms, termination, etc.)
- Make the contract legally sound and enforceable
- Add placeholder text [TO BE FILLED] for missing specific details

**RESPONSE FORMAT:**
Provide ONLY the complete contract text without any additional commentary or explanations.

Draft the contract now:
"""

CONTRACT_MODIFICATION_PROMPT = """
You are an expert contract attorney. A client wants to modify an existing contract.

**CURRENT CONTRACT:**
{current_contract}

**CLIENT'S MODIFICATION REQUEST:**
{modification_request}

**INSTRUCTIONS:**
1. Review the current contract carefully
2. Understand what changes the client wants to make
3. Apply the requested modifications to the contract
4. Ensure the modified contract remains legally sound
5. Maintain proper legal formatting and structure
6. If the modification conflicts with existing clauses, resolve appropriately
7. Add, remove, or modify clauses as requested

**MODIFICATION TYPES TO HANDLE:**
- Adding new clauses or provisions
- Modifying existing terms (payment, duration, obligations)
- Removing or replacing clauses
- Changing party information
- Updating dates, amounts, or specifications
- Adding special conditions or requirements

**RESPONSE FORMAT:**
Provide ONLY the complete modified contract text without any additional commentary.

Generate the modified contract now:
"""

CONTRACT_TYPE_DETECTION_PROMPT = """
Analyze the following contract drafting request and determine the contract type:

**REQUEST:** {query}

**INSTRUCTIONS:**
Respond with ONLY the contract type (e.g., "Lease Agreement", "Employment Contract", "Service Agreement", "Purchase Agreement", "Non-Disclosure Agreement", etc.)

Contract Type:
"""

# ========== Helper Functions ========== #
def get_gemini_response(prompt):
    """Get response from Gemini AI"""
    try:
        response = model.generate_content(prompt)
        return {"success": True, "response": response.text}
    except Exception as e:
        return {"success": False, "error": str(e)}

def check_question_relevance(question):
    """Check if the question is related to contracts using AI with strict filtering"""
    try:
        # First, check for obvious non-contract terms
        non_contract_phrases = [
            "hi", "hello", "hey", "how are you", "good morning", "good afternoon", 
            "good evening", "what's up", "how's it going", "nice to meet you",
            "weather", "food", "music", "movie", "sports", "game", "fun",
            "tell me about yourself", "who are you", "what can you do",
            "joke", "story", "recipe", "travel", "vacation", "hobby"
        ]
        
        question_lower = question.lower().strip()
        
        # Quick rejection for obvious non-contract queries - but be more careful
        if len([phrase for phrase in non_contract_phrases if phrase in question_lower]) > 0:
            # Check if it also contains contract terms - if so, don't reject
            contract_indicators = ["contract", "agreement", "lease", "landlord", "tenant", "party", "clause"]
            if not any(indicator in question_lower for indicator in contract_indicators):
                return False
        
        # Enhanced contract keywords - including rent, payment, terms, etc.
        contract_keywords = [
            "contract", "agreement", "clause", "term", "provision", "party", "parties",
            "obligation", "liability", "breach", "termination", "renewal", "payment",
            "consideration", "warranty", "guarantee", "indemnity", "confidentiality",
            "employment", "service", "sale", "purchase", "lease", "rental", "rent",
            "price", "cost", "fee", "amount", "deposit", "penalty", "damages",
            "duration", "period", "deadline", "date", "schedule", "timeline",
            "property", "premises", "landlord", "tenant", "lessee", "lessor",
            "employer", "employee", "client", "contractor", "vendor", "supplier",
            "legal", "law", "regulation", "compliance", "jurisdiction", "governing",
            "arbitration", "mediation", "dispute", "resolution", "force majeure",
            "assignment", "subcontracting", "performance", "delivery", "acceptance",
            "intellectual property", "copyright", "trademark", "patent", "royalty",
            "scope", "specification", "requirement", "milestone", "deliverable"
        ]
        
        # Check if question contains contract-related keywords
        if any(keyword in question_lower for keyword in contract_keywords):
            return True
        
        # Check for modification keywords combined with contract context
        modification_keywords = ["add", "remove", "change", "modify", "update", "edit", "include", "insert", "delete", "replace"]
        if any(mod_keyword in question_lower for mod_keyword in modification_keywords):
            # If it's a modification request and mentions contract context, it's relevant
            contract_context = ["contract", "agreement", "lease", "my contract", "the contract", "this contract"]
            if any(context in question_lower for context in contract_context):
                return True
        
        # Check for question patterns about contract content
        contract_question_patterns = [
            "what is the", "what are the", "how much", "when is", "when does",
            "where is", "who is", "which", "explain", "clarify", "interpret",
            "in this agreement", "in this contract", "in the contract", "in the agreement",
            "according to", "based on", "under this", "terms of", "conditions of",
            "my contract", "the contract", "this contract"
        ]
        
        if any(pattern in question_lower for pattern in contract_question_patterns):
            return True
        
        # Check for analysis keywords
        analysis_keywords = ["analyze", "review", "summary", "risks", "analysis", "breakdown", "examine"]
        if any(analysis_keyword in question_lower for analysis_keyword in analysis_keywords):
            contract_context = ["contract", "agreement", "lease", "this", "my"]
            if any(context in question_lower for context in contract_context):
                return True
        
        # Use AI for more sophisticated checking only if keyword checks are inconclusive
        prompt = CONTRACT_RELEVANCE_CHECK_PROMPT.format(question=question)
        result = get_gemini_response(prompt)
        
        if result["success"]:
            response = result["response"].strip().upper()
            return "RELEVANT" in response
        else:
            # Final fallback - if it mentions agreement/contract and asks about something, it's likely relevant
            contract_mentions = ["agreement", "contract", "lease", "landlord", "tenant", "party"]
            return any(mention in question_lower for mention in contract_mentions)
            
    except Exception as e:
        # Enhanced fallback logic
        question_lower = question.lower()
        
        # If it's asking about specific contract terms, it's relevant
        contract_terms = ["rent", "payment", "price", "fee", "cost", "deposit", "term", "duration", "party", "obligation", "landlord", "tenant"]
        if any(term in question_lower for term in contract_terms):
            return True
            
        # Basic contract keywords
        basic_keywords = ["contract", "agreement", "lease", "employment", "service", "purchase", "legal", "modify", "add", "change"]
        return any(keyword in question_lower for keyword in basic_keywords)
    
def parse_contract_analysis_response(response_text):

    """Parse the AI response to extract structured data"""
    try:
        # Extract key clauses using regex
        key_clauses_match = re.search(r'\*\*Key Clauses\*\*:?\s*(.*?)(?=\*\*|$)', response_text, re.DOTALL | re.IGNORECASE)
        key_clauses = []
        if key_clauses_match:
            clauses_text = key_clauses_match.group(1)
            clauses = re.findall(r'[-•*]\s*([^\n]+)', clauses_text)
            key_clauses = [clause.strip() for clause in clauses if clause.strip()]
        
        # Extract contract details
        contract_details = {}
        
        # Look for contract type
        contract_type_match = re.search(r'(?:Contract Type)[:]\s*([^\n]+)', response_text, re.IGNORECASE)
        if contract_type_match:
            contract_details['contract_type'] = contract_type_match.group(1).strip()
        
        # Look for parties
        parties_match = re.search(r'(?:Parties|Party 1)[:]\s*([^\n]+)', response_text, re.IGNORECASE)
        if parties_match:
            contract_details['parties'] = parties_match.group(1).strip()
        
        # Look for duration
        duration_match = re.search(r'(?:Duration|Term)[:]\s*([^\n]+)', response_text, re.IGNORECASE)
        if duration_match:
            contract_details['duration'] = duration_match.group(1).strip()
        
        return {
            'analysis': response_text,
            'key_clauses': key_clauses,
            'contract_details': contract_details
        }
    except Exception as e:
        return {
            'analysis': response_text,
            'key_clauses': [],
            'contract_details': {}
        }

# ========== SINGLE UNIFIED CONTRACT ENDPOINT ========== #
@app.post("/api/contract", response_model=UnifiedContractResponse, tags=["Unified Contract API"])
async def unified_contract_endpoint(request: UnifiedContractRequest):
    """
    **🎯 INTELLIGENT CONTRACT API - NO ACTION REQUIRED**
    
    Simply provide your query in natural language and the AI will understand what you want!
    
    **Examples:**
    
    **Draft Contracts:**
    ```json
    {
        "query": "Draft a lease agreement for a 2-bedroom apartment for $1500/month"
    }
    ```
    
    **Ask Questions (with session_id):**
    ```json
    {
        "query": "What are the payment terms in this contract?",
        "session_id": "your-session-id"
    }
    ```
    
    **Modify Contracts (with session_id):**
    ```json
    {
        "query": "Add a pet deposit clause of $300",
        "session_id": "your-session-id"
    }
    ```
    
    **Analyze Contracts (with session_id):**
    ```json
    {
        "query": "Analyze this contract and show me the risks",
        "session_id": "your-session-id"
    }
    ```
    
    The AI automatically detects your intent and responds accordingly!
    """
    try:
        if not request.query:
            return UnifiedContractResponse(
                success=False,
                query="",
                error="Query is required"
            )
        
        # Check if query is contract-related first
        is_relevant = check_question_relevance(request.query)
        if not is_relevant:
            return UnifiedContractResponse(
                success=False,
                query=request.query,
                error="I am a specialized contract drafting tool. I can only help with contract-related requests such as drafting contracts, analyzing agreements, or answering legal questions about contracts."
            )
        
        # Detect user intent based on query
        has_session = request.session_id and request.session_id in contract_sessions
        intent_prompt = QUERY_INTENT_DETECTION_PROMPT.format(
            query=request.query,
            has_session=has_session
        )
        intent_result = get_gemini_response(intent_prompt)
        
        if not intent_result["success"]:
            return UnifiedContractResponse(
                success=False,
                query=request.query,
                error="Error understanding your request. Please try again."
            )
        
        detected_intent = intent_result["response"].strip().upper()
        
        # Handle INVALID intent
        if detected_intent == "INVALID":
            return UnifiedContractResponse(
                success=False,
                query=request.query,
                detected_intent="INVALID",
                error="I couldn't understand your request. Please ask me to draft a contract, ask questions about a contract, modify a contract, or analyze a contract."
            )
        
        # Handle DRAFT intent
        if detected_intent == "DRAFT":
            # Additional check for drafting keywords
            drafting_keywords = ["draft", "create", "make", "write", "generate", "contract", "agreement"]
            if not any(keyword.lower() in request.query.lower() for keyword in drafting_keywords):
                return UnifiedContractResponse(
                    success=False,
                    query=request.query,
                    detected_intent="DRAFT",
                    error="Please ask me to draft or create a specific type of contract."
                )
            
            # Generate session ID and draft contract
            session_id = str(uuid.uuid4())
            
            # Detect contract type
            type_prompt = CONTRACT_TYPE_DETECTION_PROMPT.format(query=request.query)
            type_result = get_gemini_response(type_prompt)
            contract_type = "Contract" if not type_result["success"] else type_result["response"].strip()
            
            # Generate contract
            prompt = SIMPLE_CONTRACT_DRAFTING_PROMPT.format(query=request.query)
            result = get_gemini_response(prompt)
            
            if result["success"]:
                # Store in session
                contract_sessions[session_id] = {
                    "contract_text": result["response"],
                    "contract_type": contract_type,
                    "original_query": request.query,
                    "creation_time": str(uuid.uuid1().time),
                    "status": "drafted",
                    "modification_history": []
                }
                
                return UnifiedContractResponse(
                    success=True,
                    query=request.query,
                    session_id=session_id,
                    contract_text=result["response"],
                    contract_type=contract_type,
                    detected_intent="DRAFT"
                )
            else:
                return UnifiedContractResponse(
                    success=False,
                    query=request.query,
                    detected_intent="DRAFT",
                    error=f"Error generating contract: {result['error']}"
                )
        
        # For all other intents, session_id is required
        if not request.session_id:
            return UnifiedContractResponse(
                success=False,
                query=request.query,
                detected_intent=detected_intent,
                error="Session ID is required for this operation. Please draft a contract first to get a session ID."
            )
        
        # Validate session
        if request.session_id not in contract_sessions:
            return UnifiedContractResponse(
                success=False,
                query=request.query,
                session_id=request.session_id,
                detected_intent=detected_intent,
                error="Session not found. Please draft a contract first."
            )
        
        # Handle QUESTION intent
        if detected_intent == "QUESTION":
            # Get contract and answer question
            contract_data = contract_sessions[request.session_id]
            contract_text = contract_data["contract_text"]
            trimmed_text = contract_text[:20000]
            
            prompt = CONTRACT_QA_PROMPT.format(
                contract_text=trimmed_text,
                question=request.query
            )
            result = get_gemini_response(prompt)
            
            if result["success"]:
                return UnifiedContractResponse(
                    success=True,
                    query=request.query,
                    session_id=request.session_id,
                    answer=result["response"],
                    is_relevant=True,
                    detected_intent="QUESTION"
                )
            else:
                return UnifiedContractResponse(
                    success=False,
                    query=request.query,
                    session_id=request.session_id,
                    detected_intent="QUESTION",
                    error=f"Error processing question: {result['error']}"
                )
        
        # Handle MODIFY intent
        elif detected_intent == "MODIFY":
            # Get current contract and modify
            session_data = contract_sessions[request.session_id]
            current_contract = session_data["contract_text"]
            
            prompt = CONTRACT_MODIFICATION_PROMPT.format(
                current_contract=current_contract,
                modification_request=request.query
            )
            result = get_gemini_response(prompt)
            
            if result["success"]:
                # Update session
                contract_sessions[request.session_id]["contract_text"] = result["response"]
                contract_sessions[request.session_id]["modification_history"].append({
                    "request": request.query,
                    "timestamp": str(uuid.uuid1().time)
                })
                
                return UnifiedContractResponse(
                    success=True,
                    query=request.query,
                    session_id=request.session_id,
                    contract_text=result["response"],
                    modification_history=contract_sessions[request.session_id]["modification_history"],
                    detected_intent="MODIFY"
                )
            else:
                return UnifiedContractResponse(
                    success=False,
                    query=request.query,
                    session_id=request.session_id,
                    detected_intent="MODIFY",
                    error=f"Error modifying contract: {result['error']}"
                )
        
        # Handle ANALYZE intent
        elif detected_intent == "ANALYZE":
            # Get contract and analyze
            contract_data = contract_sessions[request.session_id]  
            contract_text = contract_data["contract_text"]
            trimmed_text = contract_text[:20000]
            
            prompt = CONTRACT_ANALYSIS_PROMPT.format(contract_text=trimmed_text)
            result = get_gemini_response(prompt)
            
            if result["success"]:
                parsed_data = parse_contract_analysis_response(result["response"])
                
                return UnifiedContractResponse(
                    success=True,
                    query=request.query,
                    session_id=request.session_id,
                    contract_analysis=parsed_data['analysis'],
                    key_clauses=parsed_data['key_clauses'],
                    contract_details=parsed_data['contract_details'],
                    detected_intent="ANALYZE"
                )
            else:
                return UnifiedContractResponse(
                    success=False,
                    query=request.query,
                    session_id=request.session_id,
                    detected_intent="ANALYZE",
                    error=f"Error analyzing contract: {result['error']}"
                )
        
        # Unknown intent
        else:
            return UnifiedContractResponse(
                success=False,
                query=request.query,
                detected_intent=detected_intent,
                error="I couldn't understand what you want to do. Please ask me to draft, modify, analyze, or ask questions about contracts."
            )
    
    except Exception as e:
        return UnifiedContractResponse(
            success=False,
            query=request.query,
            error=f"Unexpected error: {str(e)}"
        )

# ========== Basic Endpoints ========== #
@app.get("/", tags=["Info"])
async def root():
    """API Information and Usage Guide"""
    return {
        "message": "🎯 Intelligent Contract API",
        "description": "Just provide your query - AI understands what you want!",
        "main_endpoint": "/api/contract",
        "how_it_works": "Send natural language queries and the AI will automatically detect if you want to draft, modify, analyze, or ask questions about contracts",
        "examples": {
            "draft": {
                "query": "Draft a lease agreement for a 2-bedroom apartment"
            },
            "modify": {
                "query": "Add a pet deposit clause",
                "session_id": "your-session-id"
            },
            "question": {
                "query": "What are the payment terms?",
                "session_id": "your-session-id"
            },
            "analyze": {
                "query": "Analyze this contract and show risks",
                "session_id": "your-session-id"
            }
        },
        "workflow": [
            "1. Send any contract-related query",
            "2. AI detects your intent automatically",
            "3. Get appropriate response based on what you asked",
            "4. Use session_id for follow-up operations"
        ],
        "documentation": "/docs"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "Unified Contract Drafting API",
        "active_sessions": len(contract_sessions)
    }

# ========== Main Entry Point ========== #
if __name__ == "__main__":
    uvicorn.run("contract:app", host="0.0.0.0", port=8000, reload=True)
