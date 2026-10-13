import io
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from src.model import get_model

app = FastAPI(title="CIFAR-10 Image Classifier API")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG"}

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load trained model
MODEL_PATH = Path(__file__).resolve().parents[1] / "model.pt"
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Trained model checkpoint not found: {MODEL_PATH}")

model = get_model(num_classes=10)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
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
    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the maximum size of {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB",
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as uploaded_image:
                if uploaded_image.format not in ALLOWED_IMAGE_FORMATS:
                    raise HTTPException(
                        status_code=415,
                        detail="Unsupported image format. Upload a PNG or JPEG image.",
                    )
                uploaded_image.verify()

            with Image.open(io.BytesIO(image_bytes)) as uploaded_image:
                uploaded_image.load()
                image = uploaded_image.convert("RGB")
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image file",
        ) from exc

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
