from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import uvicorn
import io

from . import ocr_engine, ie_engine

app = FastAPI(title="Extraction Service", description="Extracts text and entities from medical images.")

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    ocr_engine.init_ocr()
    # ie_engine.init_model()
    ie_engine.init_gemini()

@app.post("/extract")
async def extract_endpoint(file: UploadFile = File(...), use_gemini: bool = Form(...), use_ollama: bool = Form(...)):
    """
    Endpoint to accept an image file, extract text, and identify entities.
    Optional 'use_gemini' flag to use Google's Gemini API for extraction.
    Optional 'use_ollama' flag to use local Ollama model (default: llama3.2).
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    try:
        # Read file content
        contents = await file.read()
        print(f"Use Gemini: {use_gemini}, Use Ollama: {use_ollama}")
        
        # Step 1: OCR
        raw_text = ocr_engine.extract_text(contents)
        
        # Step 2: IE
        if use_gemini:
             entities = ie_engine.extract_with_gemini(raw_text)
        elif use_ollama:
             entities = ie_engine.extract_with_ollama(raw_text)
        else:
             entities = ie_engine.extract_entities(raw_text)
        
        return {
            "filename": file.filename,
            "raw_text": raw_text,
            "entities": entities
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
