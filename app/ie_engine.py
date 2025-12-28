from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch
import google.generativeai as genai
import os
import json
import ollama
from dotenv import load_dotenv

def classify_document_ollama(text: str, model_name: str = "llama3") -> str:
    """Classifies the document type using Ollama."""
    
    # Simple prompt for classification
    prompt = f"""
    Classify the following medical text into exactly one of these categories:
    - Medical Report
    - Lab Report
    - Discharge Summary
    - Admission Slip

    Return ONLY the category name, nothing else.
    
    Text:
    {text[:1000]}
    """
    
    try:
        response = ollama.chat(model=model_name, messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ])
        category = response['message']['content'].strip()
        
        # Basic validation/fallback
        valid_categories = ["Medical Report", "Lab Report", "Discharge Summary", "Admission Slip"]
        
        # Simple fuzzy match or check
        for vc in valid_categories:
            if vc.lower() in category.lower():
                return vc
                
        return "Medical Report" # Default fallback
    except Exception as e:
        print(f"Ollama Classification failed: {e}")
        return "Medical Report"

def extract_with_ollama(text: str, model_name: str = "llama3") -> dict:
    """
    Extracts structured entities from medical text using Ollama (local LLM).
    Performs 'Classify & Extract' workflow.
    """
    # Step 1: Classify
    doc_type = classify_document_ollama(text, model_name)
    print(f"Detected Document Type (Ollama): {doc_type}")
    
    # Step 2: Select Prompt
    prompt_template = PROMPTS.get(doc_type, PROMPTS["Medical Report"])
    
    # Step 3: Extract
    final_prompt = f"""
    {prompt_template}
    
    Current Input Text:
    {text}
    
    CRITICAL INSTRUCTIONS:
    - Return ONLY valid JSON with proper syntax
    - Use double quotes for all strings
    - Ensure all commas and brackets are properly placed
    - Do NOT add markdown code blocks, explanations, or any text outside the JSON
    - Start your response with {{ and end with }}
    """

    try:
        response = ollama.chat(
            model=model_name, 
            messages=[
                {
                    'role': 'user',
                    'content': final_prompt,
                },
            ],
            format='json'  # Request JSON format explicitly
        )
        content = response['message']['content']
        
        # More aggressive cleaning
        content = content.strip()
        
        # Remove markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0] if content.count("```") >= 2 else content
        
        # Remove any leading/trailing whitespace
        content = content.strip()
        
        # Find JSON object boundaries
        start = content.find('{')
        end = content.rfind('}')
        
        if start != -1 and end != -1:
            content = content[start:end+1]
        
        print(f"Cleaned Ollama response: {content[:200]}...")  # Debug output
        
        return json.loads(content)
    
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw content: {content}")
        return {"error": f"Invalid JSON from model: {str(e)}", "raw_response": content[:500]}
    except Exception as e:
        print(f"Error during Ollama extraction: {e}")
        return {"error": str(e)}

# Load environment variables
load_dotenv()

# Global pipeline instance
ner_pipeline = None

# Model checkpoint - using a popular biomedical NER model
# d4data/biomedical-ner-all is a good general purpose biomedical NER model
# MODEL_CHECKPOINT = "d4data/biomedical-ner-all"
# MODEL_CHECKPOINT = "Xenova/ClinicalBERT"
MODEL_CHECKPOINT = "cp229/Bio_ClinicalBERT"

def init_gemini():
    """Initializes the Gemini API client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY not found in environment variables.")
        return False
    
    genai.configure(api_key=api_key)
    return True

def init_model():
    """Initializes the BERT NER pipeline."""
    global ner_pipeline
    if ner_pipeline is None:
        print(f"Loading NER model: {MODEL_CHECKPOINT}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
            model = AutoModelForTokenClassification.from_pretrained(MODEL_CHECKPOINT)
            
            # aggregation_strategy="simple" groups sub-words into whole words
            ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="first")
            print("NER model loaded.")
        except Exception as e:
            print(f"Failed to load NER model: {e}")
            raise e

def extract_entities(text: str) -> dict:
    """
    Extracts entities from text using the loaded NER model.
    Returns a dictionary of entities grouped by type.
    """
    global ner_pipeline
    if ner_pipeline is None:
        init_model()
    
    if not text:
        return {}

    try:
        results = ner_pipeline(text)
        
        entities = {}
        for item in results:
            entity_type = item['entity_group']
            word = item['word']
            
            if entity_type not in entities:
                entities[entity_type] = []
            
            entities[entity_type].append(word)
            
        return entities
    except Exception as e:
        print(f"Error during entity extraction: {e}")
        return {"error": str(e)}

# Prompt Library
PROMPTS = {
    "Medical Report": """
    You are an expert clinical AI assistant. Extract structured info from this Medical Report.
    
    Extract the following entities:
    - **PII**: Patient Name, DOB, Patient ID, Date of Report.
    - **Disease_disorder**: Any medical condition, illness, disease, or diagnosis.
    - **Symptoms**: Any symptoms or complaints.
    - **Clinical_finding_names**: Any clinical finding names.
    - **Clinical_finding_values**: Any clinical finding values.
    - **Medication**: Any drug or pharmaceutical product.
    - **Dosage**: The strength, frequency, or amount of a medication.
    - **Procedure**: Any medical test, surgery, or diagnostic procedure.

    Output JSON format:
    {
      "Document_Type": "Medical Report",
      "PII": {"Name": "...", "DOB": "...", "ID": "...", "Date": "..."},
      "Disease_disorder": ["..."],
      "Symptoms": ["..."],
      "Clinical_finding_names": ["..."],
      "Clinical_finding_values": ["..."],
      "Medication": ["..."],
      "Dosage": ["..."],
      "Procedure": ["..."]
    }
    """,
    "Lab Report": """
    You are an expert clinical AI assistant. Extract structured info from this Lab Report.
    
    Extract the following entities:
    - **PII**: Patient Name, DOB, Patient ID, Date of Report.
    - **Lab_Tests**: List of objects with Name, Value, Unit, Reference_Range.
    
    Output JSON format:
    {
      "Document_Type": "Lab Report",
      "PII": {"Name": "...", "DOB": "...", "ID": "...", "Date": "..."},
      "Lab_Tests": [{"Name": "...", "Value": "...", "Unit": "...", "Reference_Range": "..."}]
    }
    """,
    "Discharge Summary": """
    You are an expert clinical AI assistant. Extract structured info from this Discharge Summary.
    
    Extract the following entities:
    - **PII**: Patient Name, DOB, Patient ID, Admission Date, Discharge Date.
    - **Diagnosis**: Admission diagnosis and Discharge diagnosis.
    - **Outcome**: Clinical outcome (e.g., Improved, Stable, Deceased).
    - **Instructions**: Discharge instructions and follow-up plan.
    
    Output JSON format:
    {
      "Document_Type": "Discharge Summary",
      "PII": {"Name": "...", "DOB": "...", "ID": "...", "Admission_Date": "...", "Discharge_Date": "..."},
      "Diagnosis": ["..."],
      "Outcome": "...",
      "Instructions": ["..."]
    }
    """,
    "Admission Slip": """
    You are an expert clinical AI assistant. Extract structured info from this Admission Slip.
    
    Extract the following entities:
    - **PII**: Patient Name, DOB, Patient ID, Date of Admission.
    - **Admission_Reason**: Reason for admission or Chief Complaint.
    - **Doctor**: Admitting Physician name.
    - **Department**: Medical Department/Ward.
    
    Output JSON format:
    {
      "Document_Type": "Admission Slip",
      "PII": {"Name": "...", "DOB": "...", "ID": "...", "Date": "..."},
      "Admission_Reason": "...",
      "Doctor": "...",
      "Department": "..."
    }
    """
}

def classify_document(text: str) -> str:
    """Classifies the document type using Gemini."""
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Simple prompt for classification
    prompt = f"""
    Classify the following medical text into exactly one of these categories:
    - Medical Report
    - Lab Report
    - Discharge Summary
    - Admission Slip

    Return ONLY the category name, nothing else.
    
    Text:
    {text[:1000]} # sending first 1000 chars is usually enough for classification
    """
    
    try:
        response = model.generate_content(prompt)
        category = response.text.strip()
        
        # Basic validation/fallback
        valid_categories = ["Medical Report", "Lab Report", "Discharge Summary", "Admission Slip"]
        
        # Simple fuzzy match or check
        for vc in valid_categories:
            if vc.lower() in category.lower():
                return vc
                
        return "Medical Report" # Default fallback
    except Exception as e:
        print(f"Classification failed: {e}")
        return "Medical Report"

def extract_with_gemini(text: str) -> dict:
    """
    Extracts structured entities from medical text using Gemini API with dynamic prompting.
    """
    if not init_gemini():
        return {"error": "Gemini API key not configured."}

    # Step 1: Classify
    doc_type = classify_document(text)
    print(f"Detected Document Type: {doc_type}")
    
    # Step 2: Select Prompt
    prompt_template = PROMPTS.get(doc_type, PROMPTS["Medical Report"])
    
    # Step 3: Extract
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    final_prompt = f"""
    {prompt_template}
    
    Current Input Text:
    {text}
    """

    try:
        response = model.generate_content(final_prompt)
        content = response.text
        
        # Clean up code blocks if present
        if "```json" in content:
            content = content.replace("```json", "").replace("```", "")
        elif "```" in content:
             content = content.replace("```", "")
             
        return json.loads(content.strip())
    
    except Exception as e:
        print(f"Error during Gemini extraction: {e}")
        return {"error": str(e)}
