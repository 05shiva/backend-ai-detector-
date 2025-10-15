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

MODEL_NAME = "distilroberta-base"
tokenizer = None
model = None
device = torch.device("cpu")

@app.on_event("startup")
async def load_model():
    global tokenizer, model
    print("Loading lightweight model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    )
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    model.to(device)
    model.eval()

@app.post("/detect/text")
async def detect_text(text: str = Form(...)):
    try:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        label = "AI" if torch.argmax(probs) == 1 else "Human"
        confidence = round(float(torch.max(probs)) * 100, 2)
        return {"label": label, "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}
