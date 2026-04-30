from ocr.ocr_utils import extract_text
from nlp.text_model import predict_text

def predict_meme(image_path):
    
    # Step 1: OCR
    text = extract_text(image_path)
    print("Extracted Text:", text)

    # Step 2: NLP Prediction
    score = predict_text(text)

    # Step 3: Decision
    if score > 0.5:
        result = "TOXIC"
    else:
        result = "NON-TOXIC"

    return {
        "text": text,
        "score": score,
        "result": result
    }

# Test
if __name__ == "__main__":
    output = predict_meme("test.jpg")
    print(output)
