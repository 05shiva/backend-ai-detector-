from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI(title="AI Text Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "roberta-base-openai-detector"
tokenizer = None
model = None

@app.post("/detect/text")
async def detect_text(text: str = Form(...)):
    global tokenizer, model
    try:
        if tokenizer is None or model is None:
            print("Loading model...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        label = "AI" if torch.argmax(probs) == 1 else "Human"
        confidence = round(float(torch.max(probs)) * 100, 2)
        return {"label": label, "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}
