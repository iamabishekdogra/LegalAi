import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
import re
import uuid
from datetime import datetime

# ========== Pydantic Models ========== #
class ContractDraftingRequest(BaseModel):
    question: str
    contract_id: Optional[str] = None  # For editing existing contracts

class ContractDraftingResponse(BaseModel):
    success: Optional[bool] = None
    question: Optional[str] = None
    response: Optional[str] = None
    contract_type: Optional[str] = None
    contract_id: Optional[str] = None
    contract_draft: Optional[str] = None
    is_new_contract: Optional[bool] = None
    error: Optional[str] = None
    
    model_config = {
        "exclude_none": True
    }

# ========== FastAPI App ========== #
app = FastAPI(
    title="Contract Drafting AI Assistant", 
    description="AI-powered contract drafting assistant for legal professionals"
)

# ========== Add CORS Middleware ========== #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== In-Memory Contract Storage ========== #
contract_sessions = {}

# ========== Gemini AI Setup ========== #
GEMINI_API_KEY = "AIzaSyBaYkOY_pT-mPTtsEy-MmdmqrkImtDKTds"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-pro")

# ========== Contract Drafting Keywords ========== #
CONTRACT_KEYWORDS = [
    "contract", "agreement", "clause", "term", "provision", "party", "parties",
    "consideration", "obligation", "liability", "indemnity", "warranty", "breach",
    "termination", "dispute", "arbitration", "jurisdiction", "governing law",
    "confidentiality", "nda", "non-disclosure", "employment", "service", "purchase",
    "sale", "lease", "rental", "licensing", "partnership", "merger", "acquisition",
    "intellectual property", "copyright", "trademark", "patent", "royalty",
    "payment", "invoice", "delivery", "performance", "default", "remedy",
    "force majeure", "assignment", "subcontract", "draft", "drafting", "template",
    "legal document", "binding", "enforceable", "negotiate", "negotiation"
]

# ========== Contract Drafting Keywords for New Contracts ========== #
DRAFTING_KEYWORDS = [
    "draft", "create", "write", "make", "generate", "prepare", "develop", "compose"
]

# ========== Contract Drafting Prompt Template ========== #
CONTRACT_DRAFTING_PROMPT = """
You are an expert contract drafting attorney with 20+ years of experience in drafting various types of legal contracts and agreements. You specialize in creating comprehensive, legally sound contracts that protect clients' interests.

**IMPORTANT RESTRICTION:** You ONLY respond to questions related to contract drafting, contract law, contract clauses, agreement terms, and legal document preparation. If the user asks about anything outside of contract drafting and related legal matters, you must respond with: "I'm sorry, but I can only assist with contract drafting and related legal matters. Please ask me about contracts, agreements, clauses, or legal document preparation."

**USER QUESTION:** {question}

**YOUR ROLE:**
- Provide expert guidance on contract drafting
- Suggest appropriate clauses and terms
- Explain legal implications of contract provisions
- Offer templates and examples when relevant
- Advise on best practices for contract creation
- Help with specific contract types (employment, service, sale, NDA, etc.)

**RESPONSE GUIDELINES:**
1. **Be Comprehensive:** Provide detailed, actionable advice
2. **Be Legally Accurate:** Ensure all suggestions are legally sound
3. **Be Practical:** Focus on real-world application
4. **Include Examples:** Provide sample clauses when helpful
5. **Consider Risks:** Highlight potential legal issues
6. **Be Professional:** Maintain attorney-level expertise

**RESPONSE FORMAT:**
- Start with a brief summary of what you'll address
- Provide detailed guidance with clear sections
- Include sample language/clauses when applicable
- End with key recommendations or next steps

Remember: Only respond to contract drafting related queries. For any other topics, use the restriction message.
"""

# ========== Contract Creation Prompt Template ========== #
CONTRACT_CREATION_PROMPT = """
You are an expert contract drafting attorney. The user is requesting you to draft a complete contract based on their requirements.

**USER REQUEST:** {question}

**INSTRUCTIONS:**
1. Create a complete, professional contract draft
2. Include all essential clauses and provisions
3. Use proper legal language and formatting
4. Make it legally sound and comprehensive
5. Include standard contract elements like:
   - Title
   - Parties section
   - Recitals
   - Terms and conditions
   - Payment terms (if applicable)
   - Termination clauses
   - Dispute resolution
   - Governing law
   - Signatures section

**FORMAT YOUR RESPONSE AS:**
CONTRACT DRAFT:
[Full contract text here]

EXPLANATION:
[Brief explanation of key provisions and recommendations]
"""

# ========== Contract Edit Prompt Template ========== #
CONTRACT_EDIT_PROMPT = """
You are an expert contract drafting attorney. The user wants to modify an existing contract.

**EXISTING CONTRACT:**
{existing_contract}

**USER'S MODIFICATION REQUEST:** {question}

**INSTRUCTIONS:**
1. Review the existing contract
2. Implement the requested changes
3. Ensure legal consistency
4. Maintain proper contract structure
5. Provide the complete updated contract

**FORMAT YOUR RESPONSE AS:**
UPDATED CONTRACT DRAFT:
[Full updated contract text here]

CHANGES MADE:
[List of specific changes implemented]

EXPLANATION:
[Brief explanation of the modifications and their implications]
"""

# ========== Helper Functions ========== #
def is_contract_related(question: str) -> bool:
    """Check if the question is related to contract drafting"""
    question_lower = question.lower()
    
    # Check for contract-related keywords
    for keyword in CONTRACT_KEYWORDS:
        if keyword in question_lower:
            return True
    
    # Additional patterns that might indicate contract-related questions
    contract_patterns = [
        r'\b(how to|help with|assist with|draft|create|write)\b.*\b(contract|agreement|document)\b',
        r'\b(legal|law|clause|term|provision)\b',
        r'\b(negotiate|negotiation|binding|enforceable)\b',
        r'\b(template|sample|example)\b.*\b(contract|agreement)\b'
    ]
    
    for pattern in contract_patterns:
        if re.search(pattern, question_lower):
            return True
    
    return False

def is_contract_drafting_request(question: str) -> bool:
    """Check if user is requesting to draft a new contract"""
    question_lower = question.lower()
    
    # Check for drafting keywords combined with contract types
    for draft_keyword in DRAFTING_KEYWORDS:
        if draft_keyword in question_lower and any(keyword in question_lower for keyword in ["contract", "agreement", "nda", "employment", "service", "lease", "partnership"]):
            return True
    
    # Additional patterns for contract drafting requests
    drafting_patterns = [
        r'\b(draft|create|write|make|generate)\b.*\b(contract|agreement)\b',
        r'\b(need|want|help me with)\b.*\b(contract|agreement)\b.*\b(for|between)\b',
        r'\bcontract\b.*\b(for|between)\b'
    ]
    
    for pattern in drafting_patterns:
        if re.search(pattern, question_lower):
            return True
    
    return False

def get_gemini_response(prompt: str):
    """Get response from Gemini AI"""
    try:
        response = model.generate_content(prompt)
        return {"success": True, "response": response.text}
    except Exception as e:
        return {"success": False, "error": str(e)}

def detect_contract_type(question: str) -> Optional[str]:
    """Detect the type of contract being discussed"""
    question_lower = question.lower()
    
    contract_types = {
        "employment": ["employment", "job", "employee", "employer", "work", "salary", "wage"],
        "service": ["service", "consulting", "freelance", "contractor", "professional"],
        "sale": ["sale", "purchase", "buy", "sell", "goods", "product"],
        "lease": ["lease", "rent", "rental", "landlord", "tenant", "property"],
        "nda": ["nda", "non-disclosure", "confidentiality", "confidential", "secret"],
        "partnership": ["partnership", "joint venture", "collaboration", "partner"],
        "licensing": ["license", "licensing", "intellectual property", "ip", "copyright", "trademark"],
        "loan": ["loan", "lending", "borrow", "credit", "finance", "debt"],
        "vendor": ["vendor", "supplier", "procurement", "supply"],
        "merger": ["merger", "acquisition", "m&a", "buyout", "takeover"]
    }
    
    for contract_type, keywords in contract_types.items():
        if any(keyword in question_lower for keyword in keywords):
            return contract_type
    
    return None

def extract_contract_from_response(response: str) -> str:
    """Extract contract draft from AI response"""
    # Look for contract draft section
    if "CONTRACT DRAFT:" in response:
        parts = response.split("CONTRACT DRAFT:")
        if len(parts) > 1:
            contract_part = parts[1].split("EXPLANATION:")[0].strip()
            return contract_part
    elif "UPDATED CONTRACT DRAFT:" in response:
        parts = response.split("UPDATED CONTRACT DRAFT:")
        if len(parts) > 1:
            contract_part = parts[1].split("CHANGES MADE:")[0].strip()
            return contract_part
    
    return response

def extract_explanation_from_response(response: str) -> str:
    """Extract explanation from AI response"""
    if "EXPLANATION:" in response:
        parts = response.split("EXPLANATION:")
        if len(parts) > 1:
            return parts[1].strip()
    elif "CHANGES MADE:" in response:
        parts = response.split("CHANGES MADE:")
        if len(parts) > 1:
            explanation_part = parts[1].split("EXPLANATION:")
            if len(explanation_part) > 1:
                return explanation_part[1].strip()
            return explanation_part[0].strip()
    return ""

# ========== Main API Endpoint ========== #
@app.post("/api/contract-drafting", response_model=ContractDraftingResponse, tags=["Contract Drafting"])
async def contract_drafting_assistant(request: ContractDraftingRequest):
    """
    Contract Drafting AI Assistant
    
    This endpoint can:
    1. Answer general contract drafting questions
    2. Draft new contracts based on user requirements
    3. Edit existing contracts when provided with a contract_id
    
    **Examples:**
    - "Draft an employment contract for a software developer"
    - "Create an NDA between two companies"
    - "Add a termination clause to my contract" (with contract_id)
    - "What clauses should I include in an employment contract?"
    """
    try:
        # Check if question is contract-related
        if not is_contract_related(request.question):
            return ContractDraftingResponse(
                success=False,
                question=request.question,
                error="I'm sorry, but I can only assist with contract drafting and related legal matters. Please ask me about contracts, agreements, clauses, or legal document preparation."
            )
        
        # Detect contract type
        contract_type = detect_contract_type(request.question)
        
        # Check if this is editing an existing contract
        if request.contract_id and request.contract_id in contract_sessions:
            # Edit existing contract
            existing_contract = contract_sessions[request.contract_id]['contract_draft']
            prompt = CONTRACT_EDIT_PROMPT.format(
                existing_contract=existing_contract,
                question=request.question
            )
            result = get_gemini_response(prompt)
            
            if result["success"]:
                updated_contract = extract_contract_from_response(result["response"])
                explanation = extract_explanation_from_response(result["response"])
                
                # Update the stored contract
                contract_sessions[request.contract_id]['contract_draft'] = updated_contract
                contract_sessions[request.contract_id]['last_updated'] = datetime.now()
                contract_sessions[request.contract_id]['edit_history'].append({
                    'question': request.question,
                    'timestamp': datetime.now()
                })
                
                return ContractDraftingResponse(
                    success=True,
                    question=request.question,
                    response=explanation,
                    contract_type=contract_type,
                    contract_id=request.contract_id,
                    contract_draft=updated_contract,
                    is_new_contract=False
                )
            else:
                raise HTTPException(status_code=500, detail=result["error"])
        
        # Check if this is a request to draft a new contract
        elif is_contract_drafting_request(request.question):
            # Create new contract
            prompt = CONTRACT_CREATION_PROMPT.format(question=request.question)
            result = get_gemini_response(prompt)
            
            if result["success"]:
                contract_draft = extract_contract_from_response(result["response"])
                
                # Generate unique contract ID and store the contract
                contract_id = str(uuid.uuid4())
                contract_sessions[contract_id] = {
                    'contract_draft': contract_draft,
                    'contract_type': contract_type,
                    'created_at': datetime.now(),
                    'last_updated': datetime.now(),
                    'original_request': request.question,
                    'edit_history': []
                }
                
                # Create response with only the fields we want
                response_data = {
                    "contract_id": contract_id,
                    "contract_draft": contract_draft
                }
                
                return JSONResponse(content=response_data)
            else:
                raise HTTPException(status_code=500, detail=result["error"])
        
        else:
            # General contract drafting question
            prompt = CONTRACT_DRAFTING_PROMPT.format(question=request.question)
            result = get_gemini_response(prompt)
            
            if result["success"]:
                return ContractDraftingResponse(
                    success=True,
                    question=request.question,
                    response=result["response"],
                    contract_type=contract_type
                )
            else:
                raise HTTPException(status_code=500, detail=result["error"])
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/api/contract/{contract_id}", tags=["Contract Management"])
async def get_contract(contract_id: str):
    """
    Retrieve a specific contract by ID
    """
    if contract_id not in contract_sessions:
        raise HTTPException(status_code=404, detail="Contract not found")
    
    contract_data = contract_sessions[contract_id]
    return {
        "contract_id": contract_id,
        "contract_draft": contract_data['contract_draft'],
        "contract_type": contract_data['contract_type'],
        "created_at": contract_data['created_at'],
        "last_updated": contract_data['last_updated'],
        "original_request": contract_data['original_request'],
        "edit_count": len(contract_data['edit_history'])
    }

@app.get("/api/contracts", tags=["Contract Management"])
async def list_contracts():
    """
    List all stored contracts
    """
    contracts = []
    for contract_id, data in contract_sessions.items():
        contracts.append({
            "contract_id": contract_id,
            "contract_type": data['contract_type'],
            "created_at": data['created_at'],
            "last_updated": data['last_updated'],
            "original_request": data['original_request'][:100] + "..." if len(data['original_request']) > 100 else data['original_request']
        })
    
    return {"contracts": contracts, "total": len(contracts)}

@app.delete("/api/contract/{contract_id}", tags=["Contract Management"])
async def delete_contract(contract_id: str):
    """
    Delete a specific contract
    """
    if contract_id not in contract_sessions:
        raise HTTPException(status_code=404, detail="Contract not found")
    
    del contract_sessions[contract_id]
    return {"message": f"Contract {contract_id} deleted successfully"}

@app.get("/api/contract-types", tags=["Contract Information"])
async def get_supported_contract_types():
    """
    Get list of supported contract types and examples
    """
    return {
        "supported_contract_types": [
            {
                "type": "Employment Agreement",
                "description": "Contracts between employers and employees",
                "examples": ["Full-time employment", "Part-time employment", "Contractor agreements"]
            },
            {
                "type": "Service Agreement",
                "description": "Contracts for professional services",
                "examples": ["Consulting agreements", "Freelance contracts", "Professional services"]
            },
            {
                "type": "Sale Agreement",
                "description": "Contracts for buying/selling goods",
                "examples": ["Purchase agreements", "Sales contracts", "Product sales"]
            },
            {
                "type": "Lease Agreement",  
                "description": "Rental and leasing contracts",
                "examples": ["Property lease", "Equipment rental", "Vehicle lease"]
            },
            {
                "type": "Non-Disclosure Agreement (NDA)",
                "description": "Confidentiality agreements",
                "examples": ["Mutual NDA", "One-way NDA", "Employee confidentiality"]
            },
            {
                "type": "Partnership Agreement",
                "description": "Business partnership contracts",
                "examples": ["General partnership", "Limited partnership", "Joint venture"]
            },
            {
                "type": "Licensing Agreement",
                "description": "Intellectual property licensing",
                "examples": ["Software license", "Trademark license", "Patent license"]
            },
            {
                "type": "Loan Agreement",
                "description": "Financial lending contracts",
                "examples": ["Personal loan", "Business loan", "Promissory note"]
            }
        ]
    }

@app.get("/api/sample-questions", tags=["Contract Information"])
async def get_sample_questions():
    """
    Get sample questions you can ask the contract drafting assistant
    """
    return {
        "sample_questions": [
            # Contract drafting requests
            "Draft an employment contract for a software developer",
            "Create an NDA between two companies",
            "Write a service agreement for consulting services",
            "Generate a lease agreement for commercial property",
            
            # General questions
            "What clauses should I include in an employment contract?",
            "How do I write a termination clause for a service contract?",
            "What should be included in a payment terms section?",
            "How do I draft an indemnification clause?",
            
            # Editing requests (require contract_id)
            "Add a termination clause to my contract",
            "Modify the payment terms to net 30 days",
            "Include a confidentiality clause",
            "Change the jurisdiction to New York"
        ]
    }

@app.get("/", tags=["Info"])
async def root():
    """
    Contract Drafting AI Assistant - API Information
    """
    return {
        "message": "Welcome to the Contract Drafting AI Assistant",
        "description": "AI-powered assistant specialized in contract drafting and legal document preparation",
        "version": "2.0.0",
        "documentation": "/docs",
        "key_features": [
            "Draft complete contracts from scratch",
            "Edit and modify existing contracts",
            "Expert contract drafting guidance",
            "Clause suggestions and templates",
            "Legal risk assessment",
            "Contract type identification",
            "Professional legal advice"
        ],
        "api_endpoints": {
            "contract_drafting": {
                "method": "POST",
                "url": "/api/contract-drafting",
                "description": "Main contract drafting endpoint - create new contracts, edit existing ones, or ask general questions",
                "parameters": {
                    "question": "Your contract-related question or request (required)",
                    "contract_id": "Optional - provide to edit existing contract"
                },
                "request_body": {
                    "question": "string",
                    "contract_id": "string (optional)"
                },
                "response_types": {
                    "new_contract": {
                        "contract_id": "Generated UUID for the contract",
                        "contract_draft": "Full contract text"
                    },
                    "contract_edit": {
                        "success": "boolean",
                        "question": "Original question",
                        "response": "Explanation of changes made",
                        "contract_type": "Type of contract",
                        "contract_id": "Contract UUID",
                        "contract_draft": "Updated contract text",
                        "is_new_contract": "false"
                    },
                    "general_question": {
                        "success": "boolean",
                        "question": "Original question",
                        "response": "Expert guidance and advice",
                        "contract_type": "Detected contract type"
                    }
                },
                "examples": [
                    "Draft an employment contract for a software developer",
                    "Create an NDA between two companies",
                    "Add a termination clause to my contract (with contract_id)",
                    "What clauses should I include in an employment contract?"
                ]
            },
            "contract_management": {
                "get_contract": {
                    "method": "GET",
                    "url": "/api/contract/{contract_id}",
                    "description": "Retrieve a specific contract by ID with full details and metadata",
                    "parameters": {
                        "contract_id": "UUID of the contract to retrieve"
                    },
                    "response": {
                        "contract_id": "Contract UUID",
                        "contract_draft": "Full contract text",
                        "contract_type": "Type of contract",
                        "created_at": "Creation timestamp",
                        "last_updated": "Last modification timestamp",
                        "original_request": "Original user request",
                        "edit_count": "Number of edits made"
                    }
                },
                "list_contracts": {
                    "method": "GET",
                    "url": "/api/contracts",
                    "description": "List all stored contracts with summary information",
                    "response": {
                        "contracts": "Array of contract summaries",
                        "total": "Total number of contracts"
                    }
                },
                "delete_contract": {
                    "method": "DELETE",
                    "url": "/api/contract/{contract_id}",
                    "description": "Delete a specific contract permanently",
                    "parameters": {
                        "contract_id": "UUID of the contract to delete"
                    },
                    "response": {
                        "message": "Deletion confirmation message"
                    }
                }
            },
            "information_endpoints": {
                "contract_types": {
                    "method": "GET",
                    "url": "/api/contract-types",
                    "description": "Get list of supported contract types with descriptions and examples",
                    "response": {
                        "supported_contract_types": [
                            "Employment Agreement",
                            "Service Agreement", 
                            "Sale Agreement",
                            "Lease Agreement",
                            "Non-Disclosure Agreement (NDA)",
                            "Partnership Agreement",
                            "Licensing Agreement",
                            "Loan Agreement"
                        ]
                    }
                },
                "sample_questions": {
                    "method": "GET",
                    "url": "/api/sample-questions",
                    "description": "Get example questions you can ask the contract drafting assistant",
                    "response": {
                        "sample_questions": "Array of example questions for contract drafting, editing, and general advice"
                    }
                }
            },
            "system_endpoints": {
                "health_check": {
                    "method": "GET",
                    "url": "/health",
                    "description": "Check API health status and version information",
                    "response": {
                        "status": "healthy",
                        "service": "Contract Drafting AI Assistant",
                        "version": "2.0.0"
                    }
                },
                "api_documentation": {
                    "method": "GET",
                    "url": "/",
                    "description": "This endpoint - complete API documentation and usage guide",
                    "response": "Comprehensive API documentation with all endpoints and examples"
                },
                "swagger_docs": {
                    "method": "GET",
                    "url": "/docs",
                    "description": "Interactive Swagger/OpenAPI documentation"
                }
            }
        },
        "usage_guide": {
            "new_contract": {
                "description": "Create a new contract from scratch",
                "method": "POST /api/contract-drafting",
                "body": {
                    "question": "Draft an employment contract for a software developer"
                },
                "response": "Returns contract_id and full contract_draft"
            },
            "edit_contract": {
                "description": "Modify an existing contract",
                "method": "POST /api/contract-drafting",
                "body": {
                    "question": "Add a non-compete clause",
                    "contract_id": "your-contract-uuid"
                },
                "response": "Returns updated contract with explanation of changes"
            },
            "general_help": {
                "description": "Ask contract drafting questions and get expert advice",
                "method": "POST /api/contract-drafting",
                "body": {
                    "question": "What should be included in termination clauses?"
                },
                "response": "Returns expert legal guidance and recommendations"
            },
            "retrieve_contract": {
                "description": "Get full contract details by ID",
                "method": "GET /api/contract/{contract_id}",
                "response": "Returns complete contract with metadata"
            },
            "list_all_contracts": {
                "description": "View all stored contracts",
                "method": "GET /api/contracts",
                "response": "Returns array of contract summaries"
            }
        },
        "supported_contract_types": [
            "Employment Agreement - Contracts between employers and employees",
            "Service Agreement - Contracts for professional services",
            "Sale Agreement - Contracts for buying/selling goods",
            "Lease Agreement - Rental and leasing contracts",
            "Non-Disclosure Agreement (NDA) - Confidentiality agreements",
            "Partnership Agreement - Business partnership contracts",
            "Licensing Agreement - Intellectual property licensing",
            "Loan Agreement - Financial lending contracts"
        ],
        "request_examples": {
            "draft_employment_contract": {
                "url": "/api/contract-drafting",
                "method": "POST",
                "body": {
                    "question": "Draft an employment contract for a senior software engineer with remote work options"
                },
                "expected_response": {
                    "contract_id": "uuid-string",
                    "contract_draft": "**EMPLOYMENT AGREEMENT**..."
                }
            },
            "create_nda": {
                "url": "/api/contract-drafting", 
                "method": "POST",
                "body": {
                    "question": "Create a mutual NDA between TechCorp and DataSolutions for sharing confidential business information"
                },
                "expected_response": {
                    "contract_id": "uuid-string",
                    "contract_draft": "**NON-DISCLOSURE AGREEMENT**..."
                }
            },
            "edit_existing_contract": {
                "url": "/api/contract-drafting",
                "method": "POST", 
                "body": {
                    "question": "Modify the payment terms to net 30 days and add late payment penalties",
                    "contract_id": "12345-abcd-6789-efgh"
                },
                "expected_response": {
                    "success": True,
                    "response": "explanation of changes",
                    "contract_draft": "updated contract text"
                }
            },
            "get_contract_advice": {
                "url": "/api/contract-drafting",
                "method": "POST",
                "body": {
                    "question": "What are the key elements of a strong termination clause in employment contracts?"
                },
                "expected_response": {
                    "success": True,
                    "response": "detailed legal guidance"
                }
            },
            "retrieve_contract": {
                "url": "/api/contract/12345-abcd-6789-efgh",
                "method": "GET",
                "expected_response": {
                    "contract_id": "12345-abcd-6789-efgh",
                    "contract_draft": "full contract text",
                    "contract_type": "employment",
                    "created_at": "timestamp",
                    "edit_count": 3
                }
            },
            "list_contracts": {
                "url": "/api/contracts",
                "method": "GET",
                "expected_response": {
                    "contracts": ["array of contract summaries"],
                    "total": 5
                }
            },
            "get_contract_types": {
                "url": "/api/contract-types",
                "method": "GET",
                "expected_response": {
                    "supported_contract_types": ["detailed list with examples"]
                }
            },
            "get_sample_questions": {
                "url": "/api/sample-questions",
                "method": "GET",
                "expected_response": {
                    "sample_questions": ["array of example questions"]
                }
            }
        },
        "response_formats": {
            "successful_contract_creation": {
                "contract_id": "unique-uuid-string",
                "contract_draft": "complete contract text with proper legal formatting"
            },
            "successful_contract_edit": {
                "success": True,
                "question": "user's modification request",
                "response": "explanation of changes made",
                "contract_type": "employment|service|nda|etc",
                "contract_id": "contract-uuid",
                "contract_draft": "updated contract text",
                "is_new_contract": False
            },
            "general_advice_response": {
                "success": True,
                "question": "user's question",
                "response": "detailed expert guidance",
                "contract_type": "detected contract type"
            },
            "error_response": {
                "success": False,
                "question": "user's question",
                "error": "detailed error message"
            }
        },
        "important_notes": [
            "Only responds to contract drafting and legal document related queries",
            "Non-contract questions will be politely declined with appropriate message",
            "Contracts are stored in memory and will be lost on server restart",
            "Always consult with legal counsel before using generated contracts",
            "Generated contracts are templates and may need customization for specific situations",
            "Contract editing requires providing the contract_id from initial creation",
            "All timestamps are in ISO format",
            "Contract IDs are UUIDs and should be stored for future reference"
        ],
        "getting_started_guide": {
            "step_1": {
                "action": "Check API health",
                "endpoint": "GET /health",
                "purpose": "Verify API is running and accessible"
            },
            "step_2": {
                "action": "View supported contract types",
                "endpoint": "GET /api/contract-types",
                "purpose": "Understand available contract categories"
            },
            "step_3": {
                "action": "Review sample questions",
                "endpoint": "GET /api/sample-questions",
                "purpose": "Learn how to interact with the assistant"
            },
            "step_4": {
                "action": "Draft your first contract",
                "endpoint": "POST /api/contract-drafting",
                "purpose": "Create a contract and receive contract_id"
            },
            "step_5": {
                "action": "Save contract ID for editing",
                "purpose": "Use contract_id for future modifications"
            },
            "step_6": {
                "action": "Test contract editing",
                "endpoint": "POST /api/contract-drafting (with contract_id)",
                "purpose": "Modify existing contracts"
            }
        },
        "best_practices": [
            "Always save the contract_id returned when creating new contracts",
            "Use specific and detailed requests for better contract generation",
            "Include context about parties, jurisdiction, and specific requirements",
            "Test contract editing with small changes first",
            "Review generated contracts with legal professionals",
            "Use the GET /api/contract/{id} endpoint to retrieve full contract details",
            "Regularly backup important contracts as they're stored in memory only"
        ],
        "common_use_cases": [
            "Employment contract creation for new hires",
            "NDA generation for business partnerships",
            "Service agreements for freelancers and consultants", 
            "Lease agreements for property rentals",
            "Partnership agreements for business ventures",
            "Contract clause explanations and recommendations",
            "Legal document templates and examples",
            "Contract modification and updates"
        ],
        "restriction": "Only responds to contract drafting and legal document related queries"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy", 
        "service": "Contract Drafting AI Assistant",
        "version": "2.0.0"
    }

# ========== Main Entry Point ========== #
if __name__ == "__main__":
    uvicorn.run("main12:app", host="0.0.0.0", port=8000, reload=True)
