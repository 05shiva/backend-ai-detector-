from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import gc

app = FastAPI(title="AI Text Detector")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
MODEL_NAME = "prajjwal1/bert-tiny"
tokenizer = None
model = None
device = torch.device("cpu")

# -------------------------
# Startup event: load model
# -------------------------
@app.on_event("startup")
async def load_model():
    global tokenizer, model
    print("Loading lightweight model...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    
    # Quantize to reduce memory
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

# -------------------------
# Utility: clean text
# -------------------------
def clean_text(text: str) -> str:
    text = text.strip()
    text = " ".join(text.split())  # remove extra spaces/newlines
    return text

# -------------------------
# Utility: split text into chunks
# -------------------------
def chunk_text(text: str, max_len: int = 256):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_len):
        chunks.append(" ".join(words[i:i+max_len]))
    return chunks

# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/detect/text")
async def detect_text(text: str = Form(...)):
    try:
        text = clean_text(text)
        chunks = chunk_text(text)
        
        logits_sum = torch.zeros(1, 2)
        for chunk in chunks:
            inputs = tokenizer(
                chunk, return_tensors="pt", truncation=True, max_length=256
            ).to(device)
            
            with torch.inference_mode():
                outputs = model(**inputs)
            logits_sum += outputs.logits
            
            # Clean up memory per chunk
            del inputs, outputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Average logits and calculate probabilities
        probs = torch.softmax(logits_sum, dim=1)
        ai_prob = float(probs[0][1])
        
        # Threshold tuning: adjust for tiny model
        threshold = 0.6
        label = "AI" if ai_prob > threshold else "Human"
        confidence = round(ai_prob * 100, 2)
        
        # Final memory cleanup
        del probs, logits_sum
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {"label": label, "confidence": confidence}
    
    except Exception as e:
        return {"error": str(e)}
