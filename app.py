import os
import fitz  # PyMuPDF
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import shutil
from pathlib import Path

# ========== Create upload directory ========== #
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

# ========== Pydantic Models ========== #
class ContractDraftRequest(BaseModel):
    query: str

class ContractQARequest(BaseModel):
    contract_text: str
    question: str

class ContractTextRequest(BaseModel):
    contract_text: str

# ========== FastAPI App ========== #
app = FastAPI(title="Legal Contract API", 
              description="API for drafting, analyzing, and answering questions about legal contracts")

# ========== Add CORS Middleware ========== #
# Configure CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# ========== Gemini API ========== #
GEMINI_API_KEY = "AIzaSyBaYkOY_pT-mPTtsEy-MmdmqrkImtDKTds"
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("models/gemini-1.5-pro")

# ========== Prompt Templates ========== #
CONTRACT_DRAFT_PROMPT = """
You are a legal contract drafting expert.

Based on the following user query or contract requirement, draft a complete contract in formal legal language, including all necessary clauses, obligations, timelines, penalties, termination, and applicable Indian laws or statutes.

Query:
{query}
"""

CONTRACT_ANALYSIS_PROMPT = """
You are a highly skilled legal analyst.

Analyze the following contract text and:
1. Identify missing legal clauses, mandatory statutes, or regulatory requirements based on Indian legal practices.
2. Highlight any risks, vague clauses, or inconsistencies.
3. Mention any standard sections that should be present but are absent.

Contract Text:
{text}
"""

QUESTION_PROMPT_TEMPLATE = """
You are a legal assistant helping interpret an Indian contract.

Contract Text:
{text}

User Question:
{question}

Only use the content of the contract for your answer. Keep the answer informative and precise.
"""

# ========== Helper Functions ========== #
def cleanup_temp_file(file_path: Path):
    """Remove temporary files after processing"""
    if file_path.exists():
        os.remove(file_path)

def extract_text(file_path):
    if str(file_path).lower().endswith(".pdf"):
        doc = fitz.open(file_path)
        return "\n".join([page.get_text() for page in doc])
    elif str(file_path).lower().endswith(".txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError("Only PDF and TXT formats are supported.")

def get_gemini_response(prompt):
    try:
        response = model.generate_content(prompt)
        return {"success": True, "response": response.text}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ========== API Routes ========== #
@app.post("/api/draft-contract", tags=["Contract Drafting"])
async def draft_contract(request: ContractDraftRequest):
    """
    Draft a new contract based on requirements
    
    - **query**: Description of the contract requirements
    
    Returns a complete contract in formal legal language
    """
    try:
        prompt = CONTRACT_DRAFT_PROMPT.format(query=request.query)
        result = get_gemini_response(prompt)
        
        if result["success"]:
            return {"success": True, "contract": result["response"]}
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze-contract/file", tags=["Contract Analysis"])
async def analyze_contract_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Analyze a contract from an uploaded file (PDF or TXT)
    
    - **file**: The contract file to be analyzed
    
    Returns analysis of legal clauses, risks, and missing sections
    """
    try:
        # Check file extension
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in ["pdf", "txt"]:
            raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")
        
        # Save uploaded file temporarily
        file_path = UPLOAD_FOLDER / f"{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, file_path)
        
        # Extract and analyze text
        contract_text = extract_text(file_path)
        trimmed_text = contract_text[:15000]  # Limit text length
        prompt = CONTRACT_ANALYSIS_PROMPT.format(text=trimmed_text)
        result = get_gemini_response(prompt)
        
        if result["success"]:
            return {"success": True, "analysis": result["response"]}
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze-contract/text", tags=["Contract Analysis"])
async def analyze_contract_text(request: ContractTextRequest):
    """
    Analyze contract text provided directly
    
    - **contract_text**: The full text of the contract to be analyzed
    
    Returns analysis of legal clauses, risks, and missing sections
    """
    try:
        trimmed_text = request.contract_text[:15000]
        prompt = CONTRACT_ANALYSIS_PROMPT.format(text=trimmed_text)
        result = get_gemini_response(prompt)
        
        if result["success"]:
            return {"success": True, "analysis": result["response"]}
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/contract-qa", tags=["Contract Q&A"])
async def contract_qa(request: ContractQARequest):
    """
    Ask questions about a contract
    
    - **contract_text**: The full text of the contract
    - **question**: The specific question about the contract
    
    Returns an answer based on the contract content
    """
    try:
        trimmed_text = request.contract_text[:15000]
        prompt = QUESTION_PROMPT_TEMPLATE.format(
            text=trimmed_text, 
            question=request.question
        )
        result = get_gemini_response(prompt)
        
        if result["success"]:
            return {"success": True, "answer": result["response"]}
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint with basic info
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Legal Contract API",
        "documentation": "/docs",
        "available_endpoints": [
            "/api/draft-contract",
            "/api/analyze-contract/file",
            "/api/analyze-contract/text",
            "/api/contract-qa"
        ]
    }

# ========== Main Entry Point ========== #
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
