from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI(title="AI Text Detector")

# Allow requests from any frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load open-source Roberta model (lightweight)
MODEL_NAME = "roberta-base-openai-detector"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

@app.post("/detect/text")
async def detect_text(text: str = Form(...)):
    try:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        label = "AI" if torch.argmax(probs) == 1 else "Human"
        confidence = round(float(torch.max(probs)) * 100, 2)
        return {"label": label, "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}
