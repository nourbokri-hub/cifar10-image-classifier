import hashlib
import io
import math
import os

import pandas as pd
import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

CLASS_LABELS = {
    "airplane": "Avion",
    "automobile": "Automobile",
    "bird": "Oiseau",
    "cat": "Chat",
    "deer": "Cerf",
    "dog": "Chien",
    "frog": "Grenouille",
    "horse": "Cheval",
    "ship": "Bateau",
    "truck": "Camion",
}

st.set_page_config(
    page_title="CIFAR-10 Vision AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        color-scheme: light;
        --accent: #7964b8;
        --accent-dark: #604b9c;
        --accent-soft: #f1edfa;
        --ink: #28243a;
        --muted: #716d7d;
        --line: #e9e5f0;
        --surface: #ffffff;
    }

    .stApp {
        background: #faf9fc;
        color: var(--ink);
    }

    .block-container {
        max-width: 1220px;
        padding: 2.5rem 2rem 3rem;
    }

    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: -0.025em;
    }

    [data-testid="stSidebar"] {
        background: #f5f2fa;
        border-right: 1px solid var(--line);
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--surface);
        border-color: var(--line);
        border-radius: 18px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #faf8fd;
        border: 1px dashed #c9bee2;
        border-radius: 14px;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--accent);
    }

    [data-testid="stFileUploaderDropzoneInstructions"] > div > span:first-child {
        font-size: 0;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] > div > span:first-child::after {
        color: var(--ink);
        content: "Déposez une image ici";
        font-size: 0.9rem;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] > div > span:nth-child(2) {
        font-size: 0;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] > div > span:nth-child(2)::after {
        color: var(--muted);
        content: "Taille maximale d’envoi : 200 Mio · JPG, JPEG, PNG";
        font-size: 0.82rem;
    }

    [data-testid="stFileUploaderDropzone"] > span button {
        font-size: 0;
    }

    [data-testid="stFileUploaderDropzone"] > span button::after {
        content: "Parcourir les fichiers";
        font-size: 0.875rem;
    }

    [data-testid="stAppDeployButton"] {
        display: none;
    }

    .stButton > button[kind="primary"] {
        min-height: 3rem;
        border: 0;
        border-radius: 12px;
        background: var(--accent);
        color: white;
        font-weight: 650;
        transition: background-color 120ms ease;
    }

    .stButton > button[kind="primary"]:hover {
        background: var(--accent-dark);
        color: white;
    }

    [data-testid="stProgress"] > div > div {
        background-color: var(--accent);
    }

    .hero {
        margin: 0 0 1.8rem;
        padding: 1.7rem 1.9rem;
        border: 1px solid var(--line);
        border-radius: 20px;
        background: linear-gradient(115deg, #ffffff 15%, #f5f1fb 100%);
    }

    .hero-eyebrow {
        margin: 0 0 0.35rem;
        color: var(--accent-dark);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .hero-title {
        margin: 0;
        color: var(--ink);
        font-size: clamp(2rem, 4vw, 2.8rem);
        font-weight: 760;
        letter-spacing: -0.045em;
        line-height: 1.15;
    }

    .hero-copy {
        max-width: 680px;
        margin: 0.6rem 0 0;
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.6;
    }

    .result-label {
        margin: 0.2rem 0 0.3rem;
        color: var(--muted);
        font-size: 0.88rem;
    }

    .result-class {
        margin: 0 0 1rem;
        color: var(--accent-dark);
        font-size: clamp(1.8rem, 4vw, 2.3rem);
        font-weight: 750;
        letter-spacing: -0.035em;
    }

    .confidence-row {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 1rem;
        margin: 1.2rem 0 0.45rem;
        color: var(--muted);
        font-size: 0.9rem;
    }

    .confidence-value {
        color: var(--ink);
        font-size: 1rem;
        font-weight: 700;
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 1.3rem 1rem 2rem;
        }

        .hero {
            padding: 1.25rem;
            margin-bottom: 1.2rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Informations du projet")
    st.caption("Classification d’images avec intelligence artificielle")
    st.divider()
    st.markdown("**Jeu de données**  \nCIFAR-10 · 10 classes")
    st.markdown("**Architecture**  \nResNet18")
    st.markdown("**Framework**  \nPyTorch")
    st.markdown("**Service de prédiction**  \nFastAPI")
    st.markdown("**Suivi des expériences**  \nMLflow")
    st.markdown("**Exécution**  \nDocker Compose")
    st.divider()
    st.metric("Précision sur le jeu de test", "77,92 %")
    st.caption("Résultat vérifié sur les 10 000 images du jeu de test CIFAR-10.")

st.markdown(
    """
    <header class="hero">
        <p class="hero-eyebrow">Vision par ordinateur · CIFAR-10</p>
        <h1 class="hero-title">CIFAR-10 Vision AI</h1>
        <p class="hero-copy">
            Importez une image pour obtenir une prédiction du modèle et explorer
            ses trois classes les plus probables.
        </p>
    </header>
    """,
    unsafe_allow_html=True,
)

if "image_signature" not in st.session_state:
    st.session_state.image_signature = None
    st.session_state.prediction_result = None
    st.session_state.prediction_error = None

upload_column, result_column = st.columns([1.05, 0.95], gap="large")

with upload_column:
    with st.container(border=True):
        st.subheader("Importer une image")
        st.caption(
            "Formats acceptés : JPG ou PNG · Prédiction limitée aux fichiers de 10 Mio."
        )
        uploaded_file = st.file_uploader(
            "Choisir une image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        image_bytes = uploaded_file.getvalue() if uploaded_file else None
        image_signature = (
            hashlib.sha256(image_bytes).hexdigest() if image_bytes is not None else None
        )

        if image_signature != st.session_state.image_signature:
            st.session_state.image_signature = image_signature
            st.session_state.prediction_result = None
            st.session_state.prediction_error = None

        image_is_valid = False
        if image_bytes is not None:
            if len(image_bytes) > MAX_UPLOAD_BYTES:
                st.error("Cette image dépasse la taille maximale de 10 Mio.")
            else:
                try:
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        image.verify()
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        image.load()
                        preview = image.convert("RGB")
                    st.image(
                        preview,
                        caption="Aperçu de l’image importée",
                        width="stretch",
                    )
                    image_is_valid = True
                except (
                    UnidentifiedImageError,
                    OSError,
                    Image.DecompressionBombError,
                ):
                    st.error("Le fichier ne contient pas une image JPG ou PNG valide.")

        if st.button(
            "Classer l’image",
            type="primary",
            width="stretch",
            disabled=not image_is_valid,
        ):
            st.session_state.prediction_result = None
            st.session_state.prediction_error = None
            files = {
                "file": (
                    uploaded_file.name,
                    image_bytes,
                    uploaded_file.type or "application/octet-stream",
                )
            }
            try:
                with st.spinner("Analyse de l’image en cours…"):
                    response = requests.post(API_URL, files=files, timeout=30)
                response.raise_for_status()
                result = response.json()

                if not isinstance(result, dict):
                    raise ValueError("La réponse du service est invalide.")

                prediction = result.get("prediction")
                confidence = result.get("confidence")
                top3 = result.get("top3")
                if (
                    not isinstance(prediction, str)
                    or prediction not in CLASS_LABELS
                    or isinstance(confidence, bool)
                    or not isinstance(confidence, (int, float))
                    or not math.isfinite(confidence)
                    or not 0 <= confidence <= 1
                    or not isinstance(top3, list)
                    or len(top3) != 3
                ):
                    raise ValueError("La réponse du service ne respecte pas le format attendu.")

                validated_top3 = []
                for item in top3:
                    if not isinstance(item, dict):
                        raise ValueError("La réponse du service ne respecte pas le format attendu.")
                    class_name = item.get("class")
                    probability = item.get("confidence")
                    if (
                        not isinstance(class_name, str)
                        or class_name not in CLASS_LABELS
                        or isinstance(probability, bool)
                        or not isinstance(probability, (int, float))
                        or not math.isfinite(probability)
                        or not 0 <= probability <= 1
                    ):
                        raise ValueError("La réponse du service ne respecte pas le format attendu.")
                    validated_top3.append(
                        {
                            "class": class_name,
                            "confidence": float(probability),
                        }
                    )

                st.session_state.prediction_result = {
                    "prediction": prediction,
                    "confidence": float(confidence),
                    "top3": validated_top3,
                }
            except requests.HTTPError as error:
                detail = None
                try:
                    error_body = error.response.json()
                    if isinstance(error_body, dict):
                        detail = error_body.get("detail")
                except ValueError:
                    pass
                st.session_state.prediction_error = (
                    f"Le service a refusé l’image : {detail}"
                    if detail
                    else f"Le service de prédiction a répondu avec le code "
                    f"{error.response.status_code}."
                )
            except requests.RequestException:
                st.session_state.prediction_error = (
                    "Impossible de joindre le service de prédiction. "
                    "Vérifiez que l’API est démarrée."
                )
            except ValueError as error:
                st.session_state.prediction_error = str(error)

with result_column:
    with st.container(border=True):
        st.subheader("Résultat de la classification")
        prediction_error = st.session_state.prediction_error
        prediction_result = st.session_state.prediction_result

        if prediction_error:
            st.error(prediction_error)
        elif prediction_result is None:
            st.info("Importez une image, puis lancez sa classification.")
        else:
            predicted_class = CLASS_LABELS[prediction_result["prediction"]]
            confidence = prediction_result["confidence"]

            st.markdown('<p class="result-label">Classe prédite</p>', unsafe_allow_html=True)
            st.markdown(
                f'<p class="result-class">{predicted_class}</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="confidence-row">
                    <span>Score de confiance</span>
                    <span class="confidence-value">{confidence:.1%}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(confidence)
            st.divider()
            st.markdown("#### Les trois classes les plus probables")

            chart_data = pd.DataFrame(
                [
                    {
                        "Classe": CLASS_LABELS[item["class"]],
                        "Probabilité (%)": item["confidence"] * 100,
                    }
                    for item in prediction_result["top3"]
                ]
            )
            st.bar_chart(
                chart_data,
                x="Classe",
                y="Probabilité (%)",
                horizontal=True,
                color="#7964b8",
            )

            for item in prediction_result["top3"]:
                st.markdown(
                    f"**{CLASS_LABELS[item['class']]}** — "
                    f"{item['confidence']:.1%}"
                )

st.caption("CIFAR-10 Vision AI · PyTorch · FastAPI · Streamlit")
