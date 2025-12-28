import easyocr
import numpy as np
from PIL import Image
import io
import cv2
import fitz # PyMuPDF

# Global reader instance to avoid reloading model on every request
reader = None

def init_ocr():
    """Initializes the EasyOCR reader."""
    global reader
    if reader is None:
        print("Loading EasyOCR model...")
        # using 'en' for English. Add more languages if needed.
        reader = easyocr.Reader(['en']) 
        print("EasyOCR model loaded.")

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Applies OpenCV preprocessing to improve OCR accuracy.
    Includes grayscale conversion, Gaussian blur, and adaptive thresholding.
    """
    # Convert bytes to numpy array for OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Removed Gaussian Blur and Adaptive Thresholding as they were causing blurriness/data loss
    # EasyOCR often performs better on raw grayscale or color images for high-quality inputs
    
    # Saving debug image to verify
    img_pil = Image.fromarray(gray)
    img_pil.save('test_image.png', format='PNG')
    
    return gray

def extract_text(image_bytes: bytes) -> str:
    """
    Extracts text from image bytes using EasyOCR with preprocessing.
    """
    global reader
    if reader is None:
        init_ocr()
    
    try:
        # Check for PDF signature
        if image_bytes.startswith(b'%PDF'):
            doc = fitz.open(stream=image_bytes, filetype="pdf")
            full_text = []
            for page in doc:
                pix = page.get_pixmap(dpi=300) 
                img_data = pix.tobytes("png")
                
                # Preprocess
                processed_image = preprocess_image(img_data)
                
                # detail=0 returns just the text list
                result = reader.readtext(processed_image, detail=0)
                full_text.extend(result)
            print(" ".join(full_text))
            return " ".join(full_text)

        # Preprocess the image
        processed_image = preprocess_image(image_bytes)
        
        # detail=0 returns just the text list
        result = reader.readtext(processed_image, detail=0)
        
        # Join extracted text segments
        full_text = " ".join(result)
        return full_text
    except Exception as e:
        print(f"Error during OCR: {e}")
        return ""
