import pytesseract
from PIL import Image
import easyocr

# Tesseract OCR
def extract_text_tesseract(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"Tesseract error: {e}")
        return ""

# EasyOCR
def extract_text_easyocr(image_path):
    try:
        reader = easyocr.Reader(['en'], gpu=False)
        result = reader.readtext(image_path, detail=0)
        text = ' '.join(result)
        return text.strip()
    except Exception as e:
        print(f"EasyOCR error: {e}")
        return ""

# Hybrid OCR
def extract_text(image_path):
    text = extract_text_easyocr(image_path)
    if len(text) < 5:
        text = extract_text_tesseract(image_path)
    return text
