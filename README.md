# CIFAR-10 Image Classification Project

## Project Overview

This project was developed as part of the **Deep Learning Federated Project** module.

The objective is to design, train, evaluate, and deploy a deep learning model for multi-class image classification using modern Computer Vision techniques.

The project follows a complete **end-to-end MLOps pipeline**:

**Data → Training → Evaluation → MLflow → API → UI → Docker**

---

## Dataset

* **Dataset:** CIFAR-10
* **Task:** Multi-class image classification
* **Classes:** 10
* **Image size:** 32×32 RGB
* **Total images:** 60,000

Dataset is loaded using:

```python
torchvision.datasets.CIFAR10
```

### Why CIFAR-10?

* Standard benchmark dataset
* Lightweight and fast to train
* Suitable for CNN architectures

---

## Model

* **Architecture:** ResNet18 (Transfer Learning)
* **Framework:** PyTorch
* **Loss Function:** CrossEntropyLoss
* **Optimizer:** Adam

### Why ResNet18?

* Efficient and lightweight
* Uses residual connections to avoid vanishing gradients
* Performs well on small datasets

---

## Results

- **Final Test Accuracy:** **77.92%**
- **Training epochs:** 2
- **Output:** prediction + confidence + top-3 classes

The model achieved a final test accuracy of **61.93%** after 2 epochs.  
This result represents a baseline training performance on CIFAR-10. The main objective was to build a complete and reproducible end-to-end pipeline including training, MLflow tracking, FastAPI deployment, Streamlit interface, Docker containerization, and automated tests.

Further improvements could be achieved by:
- training for more epochs
- applying stronger data augmentation
- tuning hyperparameters
- testing more advanced architectures
### Evaluation

* Accuracy
* Loss tracking with MLflow
* Model tested via API and UI

## Model Evaluation and Analysis

The model reaches **61.93% test accuracy** after 2 epochs, which is a solid baseline performance.

Some classes may be confused because CIFAR-10 images are low-resolution images of size 32×32. Similar categories such as cats and dogs or automobiles and trucks can be harder to distinguish.

This result is acceptable as a baseline and demonstrates that the full pipeline works correctly.

---

## MLOps Pipeline

This project implements a full MLOps workflow:

* **Experiment Tracking:** MLflow
* **Backend API:** FastAPI
* **Frontend UI:** Streamlit
* **Containerization:** Docker + Docker Compose
* **Testing:** Pytest

---
## MLflow Tracking

Experiments are tracked using **MLflow**, which allows monitoring and reproducibility of training runs.

The following elements are logged:

* Training loss
* Training accuracy
* Validation loss
* Validation accuracy
* Final accuracy
* Model artifact (`model.pt`)

### Run MLflow UI

To visualize experiment results:

```bash
mlflow ui
```

Then open in your browser:

```
http://127.0.0.1:5000
```

### Benefits

* Compare different runs and hyperparameters
* Track model performance over time
* Ensure reproducibility of experiments


## Project Structure

```text
cifar10-image-classifier/
├── src/                # Dataset, preprocessing, model
├── api/                # FastAPI backend
├── streamlit_app/      # Streamlit UI
├── tests/              # Pytest tests
├── docker/             # Dockerfiles
├── notebooks/          # Experiments
├── data/               # Dataset (ignored)
├── model.pt            # Trained model
├── requirements.txt
├── docker-compose.yml
├── README.md
└── .gitignore
```

---

## How to Run the Project

### Option 1 — Docker (Recommended)

```bash
docker compose up --build
```

### Access the app:

* API: http://localhost:8000/docs
* Streamlit: http://localhost:8501

---

### Option 2 — Local (without Docker)

```bash
pip install -r requirements.txt
```

Run API:

```bash
uvicorn api.main:app --reload
```

Run Streamlit:

```bash
streamlit run streamlit_app/app.py
```

---

## API

### Endpoint

`POST /predict`

### Input

* Image file (PNG/JPG)

### Output

```json
{
  "prediction": "cat",
  "confidence": 0.92,
  "top_3": ["cat", "dog", "deer"]
}
```

---

## User Interface

The Streamlit app allows:

* Upload image (drag & drop)
* Display prediction
* Show confidence score
* Show top-3 classes

---

## Testing

Tests are implemented using **pytest**.

Run tests:

```bash
pytest -v
```

Example result:

```bash
6 passed in 9.01s
```

---

## Docker Architecture

The project uses **2 services**:

* `api` → FastAPI backend
* `streamlit` → UI

Managed via:

```bash
docker-compose.yml
```

---

## Key Features

* End-to-end ML pipeline
* Model deployment with API
* Interactive UI
* Experiment tracking (MLflow)
* Containerized environment
* Automated testing

---

## Limitations

* Small dataset (CIFAR-10)
* No advanced hyperparameter tuning
* No interpretability (Grad-CAM, etc.)

---

## Future Improvements

* Add Grad-CAM visualization
* Improve model accuracy
* Add batch prediction
* Deploy on cloud (AWS / GCP)

---

## Authors

* Nour Bokri
* Farah Jamai
* Cyrine Daghfous

---

## Conclusion

This project demonstrates a complete deep learning workflow, from model development to deployment, following best practices in MLOps.

It highlights reproducibility, scalability, and practical implementation of machine learning systems.
