import os
import fitz  # PyMuPDF
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import shutil
from pathlib import Path
import re
import uuid

# ========== Create upload directory ========== #
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

# ========== In-memory storage for case sessions ========== #
case_sessions = {}

# ========== Pydantic Models ========== #
class CaseSummaryRequest(BaseModel):
    case_text: str

class CaseQARequest(BaseModel):
    session_id: str
    question: str

class CaseUploadResponse(BaseModel):
    success: bool
    session_id: str
    summary: Optional[str] = None
    key_terms: Optional[List[str]] = None
    case_details: Optional[dict] = None
    file_processed: Optional[str] = None
    error: Optional[str] = None

class CaseQAResponse(BaseModel):
    success: bool
    question: str
    answer: Optional[str] = None
    error: Optional[str] = None

# ========== FastAPI App ========== #
app = FastAPI(
    title="Legal Case Analysis API", 
    description="Upload case files and get automatic summarization with interactive Q&A"
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


# ==========  Prompt Templates ========== #
CASE_SUMMARY_PROMPT = """
You are an expert legal analyst. Analyze the following legal case document and provide a comprehensive, well-structured analysis.

**ANALYSIS REQUIREMENTS:**

## 1. CASE OVERVIEW
- **Case Title:** [Full case name with citation]
- **Court:** [Name of court and jurisdiction]
- **Date:** [Date of judgment/decision]
- **Case Type:** [Civil/Criminal/Constitutional/Administrative]
- **Judge(s):** [Name of presiding judge(s)]

## 2. PARTIES INVOLVED
- **Appellant/Petitioner:** [Name and role]
- **Respondent/Defendant:** [Name and role]
- **Other Parties:** [If any]

## 3. FACTUAL BACKGROUND
Provide a clear, chronological summary of the facts that led to this case.

## 4. LEGAL ISSUES
List the main legal questions/issues that the court addressed:
- Issue 1: [Description]
- Issue 2: [Description]
- etc.

## 5. LEGAL PROVISIONS & STATUTES
- **Primary Acts/Laws:** [List main statutes cited]
- **Specific Sections:** [Relevant sections mentioned]
- **Constitutional Provisions:** [If applicable]

## 6. PRECEDENTS CITED
- **Case 1:** [Name] - [Key principle]
- **Case 2:** [Name] - [Key principle]
- etc.

## 7. COURT'S REASONING & ANALYSIS
Explain the court's legal reasoning and analysis for each major issue.

## 8. FINAL JUDGMENT/ORDER
- **Decision:** [What the court decided]
- **Relief Granted:** [Orders passed]
- **Directions:** [Any specific directions to parties]

## 9. KEY LEGAL PRINCIPLES ESTABLISHED
List the important legal principles or precedents established by this case.

---
**CASE DOCUMENT:**
{case_text}

**Instructions:** 
- Be precise and legally accurate
- Use clear headings and bullet points
- If any information is not available, state "Not specified in the document"
- Focus on legal significance and implications
"""

CASE_QA_PROMPT = """
You are a senior legal counsel with expertise in case law analysis. A client has asked you a specific question about a legal case. Provide a comprehensive, natural response that directly addresses their question.

**CASE DOCUMENT:**
{case_text}

**CLIENT QUESTION:**
{question}

**INSTRUCTIONS FOR YOUR RESPONSE:**

1. **Answer Naturally:** Respond in a conversational, professional manner that directly addresses the specific question asked
2. **Be Comprehensive:** Include all relevant details from the case that relate to the question
3. **Use Legal Precision:** Be legally accurate and use appropriate legal terminology
4. **Include Evidence:** Reference specific parts of the case, quotes, or citations when relevant
5. **Provide Context:** Give necessary background information to make your answer complete
6. **Be Honest:** If the case doesn't contain the requested information, clearly state this
7. **Adapt Format:** Structure your response based on what the question is asking for - whether it's a simple fact, analysis, comparison, explanation, etc.

**RESPONSE GUIDELINES:**
- For factual questions: Provide direct, clear answers with supporting details
- For analytical questions: Explain the reasoning, implications, and significance
- For procedural questions: Walk through the relevant process or timeline
- For legal principle questions: Explain the concept and how it applies in this case
- For comparison questions: Compare and contrast the relevant elements
- For outcome questions: Explain the judgment, orders, and their implications

**Important:** Base your answer strictly on the provided case document. If information is not available in the case, explicitly state "This information is not provided in the case document."

Respond naturally and professionally, as if you're speaking directly to the client who asked the question.
"""

# ========== Helper Functions ========== #
def cleanup_temp_file(file_path: Path):
    """Remove temporary files after processing"""
    if file_path.exists():
        os.remove(file_path)

def extract_text_from_file(file_path):
    """Extract text from PDF or TXT files"""
    try:
        if str(file_path).lower().endswith(".pdf"):
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        elif str(file_path).lower().endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError("Only PDF and TXT formats are supported.")
    except Exception as e:
        raise ValueError(f"Error extracting text: {str(e)}")

def get_gemini_response(prompt):
    """Get response from Gemini AI"""
    try:
        response = model.generate_content(prompt)
        return {"success": True, "response": response.text}
    except Exception as e:
        return {"success": False, "error": str(e)}

def parse_case_summary_response(response_text):
    """Parse the AI response to extract structured data"""
    try:
        # Extract key terms using regex
        key_terms_match = re.search(r'\*\*Key Terms\*\*:?\s*(.*?)(?=\*\*|$)', response_text, re.DOTALL | re.IGNORECASE)
        key_terms = []
        if key_terms_match:
            terms_text = key_terms_match.group(1)
            # Extract terms from bullet points or numbered lists
            terms = re.findall(r'[-•*]\s*([^\n]+)', terms_text)
            key_terms = [term.strip() for term in terms if term.strip()]
        
        # Extract case details
        case_details = {}
        
        # Look for case name
        case_name_match = re.search(r'(?:Case name|Case Name)[:]\s*([^\n]+)', response_text, re.IGNORECASE)
        if case_name_match:
            case_details['case_name'] = case_name_match.group(1).strip()
        
        # Look for court
        court_match = re.search(r'(?:Court|Jurisdiction)[:]\s*([^\n]+)', response_text, re.IGNORECASE)
        if court_match:
            case_details['court'] = court_match.group(1).strip()
        
        # Look for date
        date_match = re.search(r'(?:Date|Date of judgment)[:]\s*([^\n]+)', response_text, re.IGNORECASE)
        if date_match:
            case_details['date'] = date_match.group(1).strip()
        
        return {
            'summary': response_text,
            'key_terms': key_terms,
            'case_details': case_details
        }
    except Exception as e:
        return {
            'summary': response_text,
            'key_terms': [],
            'case_details': {}
        }

# ========== Main Combined API Route ========== #

@app.post("/api/upload-case", response_model=CaseUploadResponse, tags=["Case Analysis"])
async def upload_and_analyze_case(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a legal case file and get automatic summarization
    
    This endpoint:
    1. Uploads and validates the case file (PDF or TXT)
    2. Extracts text from the file
    3. Automatically generates comprehensive case summary
    4. Returns a session_id for follow-up Q&A
    
    - **file**: The case document file (PDF or TXT)
    
    Returns:
    - Comprehensive case summary with key terms and case details
    - session_id for Q&A interactions
    """
    try:
        # Validate file extension
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in ["pdf", "txt"]:
            raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Save uploaded file temporarily
        file_path = UPLOAD_FOLDER / f"{session_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, file_path)
        
        # Extract text from file
        case_text = extract_text_from_file(file_path)
        
        # Store case text in session for Q&A
        case_sessions[session_id] = {
            "case_text": case_text,
            "filename": file.filename,
            "upload_time": str(uuid.uuid1().time)
        }
        
        # Limit text length for processing
        trimmed_text = case_text[:20000]
        
        # Generate summary using AI
        prompt = CASE_SUMMARY_PROMPT.format(case_text=trimmed_text)
        result = get_gemini_response(prompt)
        
        if result["success"]:
            parsed_data = parse_case_summary_response(result["response"])
            return CaseUploadResponse(
                success=True,
                session_id=session_id,
                summary=parsed_data['summary'],
                key_terms=parsed_data['key_terms'],
                case_details=parsed_data['case_details'],
                file_processed=file.filename
            )
        else:
            # Clean up session on error
            if session_id in case_sessions:
                del case_sessions[session_id]
            raise HTTPException(status_code=500, detail=result["error"])
    
    except HTTPException:
        raise
    except Exception as e:
        # Clean up session on error
        if 'session_id' in locals() and session_id in case_sessions:
            del case_sessions[session_id]
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/api/ask-question", response_model=CaseQAResponse, tags=["Case Q&A"])
async def ask_question_about_case(request: CaseQARequest):
    """
    Ask questions about an uploaded case using session_id
    
    After uploading a case with /api/upload-case, use the returned session_id
    to ask specific questions about the case content.
    
    - **session_id**: The session ID returned from case upload
    - **question**: Your specific question about the case
    
    Returns detailed answer based on the case content
    """
    try:
        # Validate session
        if request.session_id not in case_sessions:
            raise HTTPException(
                status_code=404, 
                detail="Session not found. Please upload a case file first using /api/upload-case"
            )
        
        # Get case text from session
        case_data = case_sessions[request.session_id]
        case_text = case_data["case_text"]
        
        # Limit text length for processing
        trimmed_text = case_text[:20000]
        
        # Generate Q&A response
        prompt = CASE_QA_PROMPT.format(
            case_text=trimmed_text, 
            question=request.question
        )
        result = get_gemini_response(prompt)
        
        if result["success"]:
            return CaseQAResponse(
                success=True,
                question=request.question,
                answer=result["response"]
            )
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/api/session/{session_id}", tags=["Session Management"])
async def get_session_info(session_id: str):
    """
    Get information about a case session
    
    - **session_id**: The session ID to check
    
    Returns session details and case metadata
    """
    if session_id not in case_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = case_sessions[session_id]
    return {
        "session_id": session_id,
        "filename": session_data["filename"],
        "upload_time": session_data["upload_time"],
        "case_text_length": len(session_data["case_text"]),
        "status": "active"
    }

@app.delete("/api/session/{session_id}", tags=["Session Management"])
async def delete_session(session_id: str):
    """
    Delete a case session and free up memory
    
    - **session_id**: The session ID to delete
    """
    if session_id not in case_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del case_sessions[session_id]
    return {"message": f"Session {session_id} deleted successfully"}

@app.get("/api/sessions", tags=["Session Management"])
async def list_active_sessions():
    """
    List all active case sessions
    """
    sessions = []
    for session_id, data in case_sessions.items():
        sessions.append({
            "session_id": session_id,
            "filename": data["filename"],
            "upload_time": data["upload_time"]
        })
    
    return {
        "active_sessions": len(sessions),
        "sessions": sessions
    }

@app.get("/", tags=["Info"])
async def root():
    """
    API Information and Usage Guide
    """
    return {
        "message": "Welcome to the Legal Case Analysis API",
        "description": "Upload case files and get automatic summarization with interactive Q&A",
        "documentation": "/docs",
        "usage_flow": [
            "1. Upload case file using POST /api/upload-case",
            "2. Get automatic case summary and session_id",
            "3. Ask questions using POST /api/ask-question with session_id",
            "4. Continue Q&A with the same session_id"
        ],
        "main_endpoints": [
            "POST /api/upload-case - Upload and analyze case file",
            "POST /api/ask-question - Ask questions about uploaded case",
            "GET /api/session/{session_id} - Get session info",
            "DELETE /api/session/{session_id} - Delete session"
        ],
        "supported_formats": ["PDF", "TXT"],
        "features": [
            "Automatic case summarization on upload",
            "Interactive Q&A with session management",
            "Key terms and case details extraction",
            "Memory-efficient session handling"
        ]
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy", 
        "service": "Legal Case Analysis API",
        "active_sessions": len(case_sessions)
    }

# ========== Main Entry Point ========== #
if __name__ == "__main__":
    uvicorn.run("main2:app", host="0.0.0.0", port=8000, reload=True)
