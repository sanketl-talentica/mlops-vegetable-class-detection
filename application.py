import os
import io
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from config.paths_config import MODEL_OUTPUT_PATH, CLASS_NAMES_PATH

app = FastAPI(title="Vegetable Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ImageNet normalization — must match what was used during training
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names and model only if training artifacts exist
CLASS_NAMES = None
loaded_model = None

if os.path.exists(CLASS_NAMES_PATH) and os.path.exists(MODEL_OUTPUT_PATH):
    with open(CLASS_NAMES_PATH, "r") as f:
        CLASS_NAMES = json.load(f)

    NUM_CLASSES = len(CLASS_NAMES)

    model = models.vgg16(weights=None)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_OUTPUT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    loaded_model = model

# Inference transform — same as eval transform used during training
infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


@app.get("/", response_class=HTMLResponse)
def home():
    with open("templates/index.html", "r") as f:
        return f.read()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if loaded_model is None:
        return JSONResponse(status_code=503, content={"error": "Model not ready. Run the training pipeline first."})
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = infer_transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = loaded_model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, pred_idx = torch.max(probabilities, dim=0)

        predicted_class = CLASS_NAMES[pred_idx.item()]
        confidence_pct = round(confidence.item() * 100, 2)

        # Top-3 predictions for richer response
        top3_probs, top3_idxs = torch.topk(probabilities, 3)
        top3 = [
            {"class": CLASS_NAMES[i.item()], "confidence": round(p.item() * 100, 2)}
            for p, i in zip(top3_probs, top3_idxs)
        ]

        return {
            "predicted_class": predicted_class,
            "confidence": confidence_pct,
            "top3": top3,
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
