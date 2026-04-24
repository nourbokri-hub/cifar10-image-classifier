import streamlit as st
import requests
from PIL import Image
import io
import os
import pandas as pd

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

st.set_page_config(page_title="CIFAR-10 Classifier", page_icon="🧠", layout="wide")
with st.sidebar:
    st.header("Project Info")
    st.write("**Dataset:** CIFAR-10")
    st.write("**Model:** ResNet18")
    st.write("**Framework:** PyTorch")
    st.write("**API:** FastAPI")
    st.write("**Tracking:** MLflow")
    st.write("**Docker:** docker-compose")
    st.write("**Accuracy:** 77.92%")

# --- Minimal CSS for a more "pro" look ---
st.markdown(
    """
    <style>
    .main-title {font-size: 44px; font-weight: 800; margin-bottom: 0.2rem;}
    .subtitle {font-size: 16px; opacity: 0.85; margin-top: 0;}
    .card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 18px 18px 14px 18px;
        margin-top: 12px;
    }
    .big-result {
        font-size: 28px; font-weight: 800;
        padding: 8px 0 2px 0;
    }
    .muted {opacity: 0.75;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">CIFAR-10 Image Classifier</div>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image → FastAPI predicts → Streamlit displays Top-3 with confidence.</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📤 Upload")
    uploaded = st.file_uploader("Choose an image (jpg/png)", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded is not None:
        image_bytes = uploaded.getvalue()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        st.image(img, caption="Uploaded image", use_column_width=True)


with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧠 Prediction")

    if uploaded is None:
        st.info("Upload an image to get a prediction.")
    else:
        predict_btn = st.button("🚀 Predict", use_container_width=True)

        if predict_btn:
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            try:
                with st.spinner("Running inference..."):
                  r = requests.post(API_URL, files=files, timeout=30)
                if r.status_code == 200:
                    data = r.json()

                    pred = data.get("prediction", "N/A")
                    conf = float(data.get("confidence", 0.0))

                    st.markdown(
    f"""
    <div style="
        background: linear-gradient(90deg, #4CAF50, #00C9A7);
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        font-size: 30px;
        font-weight: bold;
        color: white;">
        {pred.upper()}
    </div>
    """,
    unsafe_allow_html=True
)
                    st.markdown(f'<div class="muted">Confidence: {conf*100:.2f}%</div>', unsafe_allow_html=True)
                    st.progress(min(max(conf, 0.0), 1.0))

                    st.markdown("<hr style='border:1px solid rgba(255,255,255,0.1)'>", unsafe_allow_html=True)

                    st.markdown("### Top-3 classes")
                    top3 = data.get("top3", [])
                    df = pd.DataFrame(top3)
                    st.bar_chart(df.set_index("class"))
                    
                    for item in top3:
                        c = item["class"]
                        p = float(item["confidence"])
                        st.write(f"**{c}** — {p*100:.2f}%")
                        st.progress(min(max(p, 0.0), 1.0))

                else:
                    st.error(f"API error: {r.status_code}")
                    st.code(r.text)
            except Exception as e:
                st.error(f"Connection error: {e}")
                st.info("Make sure the API is running. Locally: uvicorn api.main:app --reload | Docker: docker compose up")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Built with PyTorch + MLflow + FastAPI + Streamlit | CIFAR-10 project")
