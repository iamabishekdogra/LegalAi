import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import re
import uuid
from datetime import datetime

# ========== Pydantic Models ========== #
class ContractDraftingRequest(BaseModel):
    question: str
    contract_id: Optional[str] = None  # For editing existing contracts

class ContractDraftingResponse(BaseModel):
    contract_id: Optional[str] = None
    contract_draft: Optional[str] = None

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

# ========== Contract Drafting Keywords for New Contracts ========== #
DRAFTING_KEYWORDS = [
    "draft", "create", "write", "make", "generate", "prepare", "develop", "compose"
]

def is_contract_drafting_request(question: str) -> bool:
    question_lower = question.lower()
    for draft_keyword in DRAFTING_KEYWORDS:
        if draft_keyword in question_lower and any(keyword in question_lower for keyword in ["contract", "agreement", "nda", "employment", "service", "lease", "partnership"]):
            return True
    return False

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
"""

def get_gemini_response(prompt: str):
    try:
        response = model.generate_content(prompt)
        return {"success": True, "response": response.text}
    except Exception as e:
        return {"success": False, "error": str(e)}

def extract_contract_from_response(response: str) -> str:
    if "CONTRACT DRAFT:" in response:
        parts = response.split("CONTRACT DRAFT:")
        if len(parts) > 1:
            return parts[1].strip()
    return response

@app.post("/api/contract-drafting", response_model=ContractDraftingResponse, tags=["Contract Drafting"])
async def contract_drafting_assistant(request: ContractDraftingRequest):
    # If editing an existing contract
    if request.contract_id and request.contract_id in contract_sessions:
        existing_contract = contract_sessions[request.contract_id]['contract_draft']
        
        # Create edit prompt with existing contract and user's modification request
        edit_prompt = f"""
You are an expert contract drafting attorney. Here is the existing contract:

{existing_contract}

The user wants to make the following change or addition:
"{request.question}"

Please update the contract accordingly by incorporating the requested changes. 
Return the complete updated contract with all modifications properly integrated.

**FORMAT YOUR RESPONSE AS:**
CONTRACT DRAFT:
[Full updated contract text here]
"""
        
        result = get_gemini_response(edit_prompt)
        if result["success"]:
            updated_contract = extract_contract_from_response(result["response"])
            # Update the stored contract
            contract_sessions[request.contract_id]['contract_draft'] = updated_contract
            contract_sessions[request.contract_id]['last_updated'] = datetime.now()
            
            return ContractDraftingResponse(
                contract_id=request.contract_id,
                contract_draft=updated_contract
            )
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    
    # If creating a new contract
    elif is_contract_drafting_request(request.question):
        prompt = CONTRACT_CREATION_PROMPT.format(question=request.question)
        result = get_gemini_response(prompt)
        if result["success"]:
            contract_draft = extract_contract_from_response(result["response"])
            contract_id = str(uuid.uuid4())
            contract_sessions[contract_id] = {
                'contract_draft': contract_draft,
                'created_at': datetime.now(),
                'last_updated': datetime.now(),
                'original_request': request.question
            }
            return ContractDraftingResponse(
                contract_id=contract_id,
                contract_draft=contract_draft
            )
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    
    # If contract_id provided but not found
    elif request.contract_id:
        raise HTTPException(status_code=404, detail="Contract not found")
    
    else:
        raise HTTPException(status_code=400, detail="Only contract drafting requests are supported.")

# ========== Main Entry Point ========== #
if __name__ == "__main__":
    uvicorn.run("main12:app", host="0.0.0.0", port=8000, reload=True)
