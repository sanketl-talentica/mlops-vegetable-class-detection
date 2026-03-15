# MLOps — Vegetable Classification (VGG16)

A production-grade MLOps pipeline for classifying vegetable images into 15 categories using **VGG16 transfer learning** (PyTorch). Built with modular pipeline stages, config-driven execution, experiment tracking, data versioning, and CI/CD deployment.

---

## Tech Stack

| Area | Tool |
|---|---|
| ML Model | VGG16 (PyTorch, pretrained on ImageNet) |
| Experiment Tracking | MLflow + DagsHub |
| Data Versioning | DVC |
| API | FastAPI |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Deployment | EC2 |

---

## Classes (7)

Broccoli, Capsicum, Carrot, Cauliflower, Potato, Radish, Tomato

Dataset: [Kaggle — Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)

---

## Project Structure

```
mlops-vegetable-class-detection/
├── config/
│   ├── config.yaml              # Pipeline config (image size, batch size, class names)
│   ├── paths_config.py          # Centralised file path constants
│   └── model_params.py          # VGG16 hyperparameters
├── dataset/
│   ├── train/                   # 15 class folders
│   ├── validation/              # 15 class folders
│   └── test/                    # 15 class folders
├── artifacts/
│   └── models/
│       ├── vgg16_model.pth      # Saved model weights
│       └── class_names.json     # Class index → label mapping
├── src/
│   ├── data_ingestion.py        # Stage 1: Validate splits, corruption/duplicate checks, save class names
│   ├── data_preprocessing.py    # Stage 2: Builds PyTorch DataLoaders with augmentation
│   ├── model_training.py        # Stage 3: Optuna tuning, train VGG16, per-class metrics, MLflow
│   ├── logger.py                # Centralised logging
│   └── custom_exception.py      # Custom exception handler
├── pipeline/
│   └── training_pipeline.py     # Runs all stages sequentially
├── utils/
│   └── common.py                # Shared utilities (read_yaml)
├── templates/
│   └── index.html               # Image upload UI
├── static/
│   └── style.css
├── application.py               # FastAPI app (image upload + prediction)
├── dvc.yaml                     # DVC pipeline definition
├── Dockerfile
├── .github/workflows/ci-cd.yml  # GitHub Actions CI/CD
└── requirements.txt
```

---

## Pipeline Stages

### Stage 1 — Data Ingestion
- Validates that `train/`, `validation/`, and `test/` directories exist and have identical class folders
- Counts images per split and logs them
- Scans for **corrupted images** (unreadable / invalid headers)
- Detects **duplicate images** across splits using MD5 hashing
- Saves `class_names.json` to `artifacts/models/` for use at inference time

### Stage 2 — Data Preprocessing
- Builds PyTorch `DataLoader`s for all three splits
- **Train augmentation** (applied only during training to improve generalisation):
  - Random horizontal flip
  - Random rotation ±15°
  - Color jitter (brightness, contrast, saturation ±0.2)
  - Resize to 224×224 + ImageNet normalization
- **Val/Test**: resize 224×224 + ImageNet normalization only — no augmentation to keep evaluation unbiased

### Stage 3 — Model Training
- Runs **Optuna hyperparameter search** (`n_trials=10`) over `learning_rate`, `batch_size`, and `epochs`
- Trains final model with best hyperparameters found
- Loads **VGG16** pretrained on ImageNet (`IMAGENET1K_V1`), backbone frozen
- Trains with **SGD** (momentum + weight decay) and **StepLR** scheduler
- Saves best model checkpoint based on validation accuracy
- Evaluates on test set: overall accuracy + **per-class precision, recall, F1** + **confusion matrix**
- Logs all params, per-epoch metrics, per-class F1, and artifacts to **MLflow + DagsHub**
- Registers model in **MLflow Model Registry** as `VegetableClassificationVGG16`
- Writes `metrics.json` for DVC metrics tracking

---

## Design Decisions

**Why VGG16?** Simple architecture, well understood, strong ImageNet pretraining. Good baseline for a 15-class image classification task.

**Why freeze the backbone?** The dataset is relatively small. Freezing the pretrained feature extractor and training only the classifier head reduces overfitting and speeds up training significantly.

**Why ImageNet normalization?** VGG16 was pretrained on ImageNet using mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`. Using the same normalization at inference and training is required for correct behaviour.

**Why accuracy as primary metric?** The vegetable dataset has balanced classes (equal number of images per class), so accuracy is a reliable metric — unlike imbalanced datasets where F1 is preferred.

**Why SGD over Adam?** SGD with momentum generalises better for fine-tuning pretrained CNNs. Adam can converge faster but often to sharper minima.

---

## Running the Pipeline

### Option 1 — DVC (recommended)
```bash
dvc repro          # runs only changed stages
dvc metrics show   # view test_acc, val_acc etc.
```

### Option 2 — Manual
```bash
python src/data_ingestion.py
python src/model_training.py    # runs preprocessing internally via DataPreprocessor
```

### Option 3 — Pipeline script
```bash
python pipeline/training_pipeline.py
```

---

## API Endpoints

Start the server:
```bash
python application.py
```

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Image upload UI |
| `/predict` | POST | Classifies uploaded image, returns class + confidence |
| `/docs` | GET | Auto-generated Swagger UI |

### Sample `/predict` request (curl)
```bash
curl -X POST http://localhost:8080/predict \
  -F "file=@tomato.jpg"
```

### Sample response
```json
{
  "predicted_class": "Tomato",
  "confidence": 97.43,
  "top3": [
    {"class": "Tomato", "confidence": 97.43},
    {"class": "Capsicum", "confidence": 1.82},
    {"class": "Brinjal", "confidence": 0.51}
  ]
}
```

---

## Docker

```bash
# Build
docker build -t vegetable-classification .

# Run
docker run -p 8080:8080 vegetable-classification
```

---

## CI/CD Pipeline (GitHub Actions)

On every push to `main`:

```
git push
    ↓
Job 1 — Train:   dvc pull → dvc repro → dvc push
    ↓
Job 2 — Build:   docker build → push to Docker Hub
    ↓
Job 3 — Deploy:  SSH into EC2 → docker pull → restart container
```

### Required GitHub Secrets

| Secret | Description |
|---|---|
| `DAGSHUB_USERNAME` | DagsHub username |
| `DAGSHUB_REPO` | DagsHub repository name |
| `DAGSHUB_TOKEN` | DagsHub access token |
| `MLFLOW_TRACKING_USERNAME` | DagsHub username |
| `MLFLOW_TRACKING_PASSWORD` | DagsHub token |
| `DOCKER_USERNAME` | Docker Hub username |
| `DOCKER_PASSWORD` | Docker Hub password |
| `EC2_HOST` | EC2 public IP |
| `EC2_USER` | EC2 username (e.g. `ubuntu`) |
| `EC2_SSH_KEY` | EC2 private key (.pem contents) |

---

## Setup

```bash
# Create virtual environment
python3 -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Download dataset from Kaggle and place in dataset/
# Structure should be: dataset/train/, dataset/validation/, dataset/test/
# Each with 15 class subfolders

# Run pipeline
dvc repro

# Start API
python application.py
```

---

## Dependencies

| Package | Purpose |
|---|---|
| torch, torchvision | VGG16 model, DataLoaders, transforms |
| Pillow | Image loading at inference |
| scikit-learn | (utilities) |
| mlflow, dagshub | Experiment tracking and model registry |
| fastapi, uvicorn | REST API server |
| python-multipart | File upload support for FastAPI |
| dvc | Data and pipeline versioning |
| python-dotenv | Environment variable management |
