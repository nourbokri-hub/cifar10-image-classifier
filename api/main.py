
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
from torchvision import transforms
import torch.nn.functional as F

from src.model import get_model

app = FastAPI(title="CIFAR-10 Image Classifier API")

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load trained model
model = get_model(num_classes=10)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.to(device)
model.eval()

# CIFAR-10 classes
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])


# 🔥 Health check (important for MLOps)
@app.get("/health")
def health():
    return {"status": "ok"}


# 🔥 Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)[0]

    # Top-3 predictions
    top_probs, top_idxs = torch.topk(probs, k=3)

    top3 = [
        {
            "class": classes[idx.item()],
            "confidence": float(top_probs[i].item())
        }
        for i, idx in enumerate(top_idxs)
    ]

    best = top3[0]

    return {
        "prediction": best["class"],
        "confidence": best["confidence"],
        "top3": top3
    }
