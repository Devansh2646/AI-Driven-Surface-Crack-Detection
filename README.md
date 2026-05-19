---
title: Steel Surface Crack Detector
colorFrom: red
colorTo: gray
sdk: docker
pinned: false
app_file: app.py
---

<h1 align="center">Steel Surface Crack Detector</h1>

<p align="center">
  Real-time CNN-powered crack detection for steel surfaces — capture, upload, and get an instant verdict with visual crack highlighting.
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/p1yushpsi/Steel_Crack_Detector">
    <img src="https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face-orange?style=for-the-badge" alt="Live Demo" />
  </a>
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11" />
  <img src="https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow 2.15" />
  <img src="https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React 18" />
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License" />
</p>

---

## What it does

Upload or capture a photo of a steel surface. The app runs it through a trained convolutional neural network and returns a **CRACK** or **NO CRACK** verdict in under a second — along with a confidence score and a red crack-region overlay showing exactly where the model found the defect.

**[→ Try the live demo on Hugging Face Spaces](https://huggingface.co/spaces/p1yushpsi/Steel_Crack_Detector)**

---

## Features

- **Live camera capture** — use your device's rear camera directly in the browser; no app install required.
- **Image upload fallback** — drag-and-drop or file-pick any JPG/PNG for batch-style inspection.
- **CNN inference** — a custom 2-block convolutional network trained on labelled steel surface images, returning a binary verdict with raw probability score.
- **Crack-region overlay** — when a crack is detected, the backend generates a red highlight over the defect region using Grad-CAM guidance + morphological contour selection. Clean surfaces get a subtle green wash instead.
- **Confidence meter** — visual progress bar showing model certainty for the returned verdict.
- **Feedback reporting** — flag incorrect verdicts; images and correction labels are stored server-side for offline dataset improvement and future retraining rounds.
- **Dockerized deployment** — single multi-stage Docker build (Node → Python); runs anywhere Docker does.

---

## Architecture

```
Browser (React + Vite)
    │
    │  POST /api/predict   (multipart image)
    │  POST /api/feedback  (multipart image + correction label)
    ▼
FastAPI  (uvicorn, port 7860)
    │
    ├── TensorFlow CNN  →  binary verdict + crack probability
    ├── Grad-CAM        →  last conv-layer activation gradients
    └── OpenCV          →  morphological crack mask + red overlay
```

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite, Phosphor Icons |
| Backend API | FastAPI, Uvicorn |
| ML inference | TensorFlow 2.15 (CPU), Keras |
| Explainability | Grad-CAM (custom implementation) |
| Image processing | OpenCV, Pillow, NumPy |
| Containerisation | Docker (multi-stage, Node + Python 3.11-slim) |
| Deployment | Hugging Face Spaces (Docker SDK) |

---

## Model

The CNN uses two convolutional blocks followed by a dense classifier:

```
Input (128×128×3)
 → Conv2D(32) + BatchNorm + MaxPool
 → Conv2D(64) + BatchNorm + MaxPool
 → Flatten
 → Dense(128, relu, dropout=0.6)
 → Dense(1, sigmoid)          ← crack probability
```

- Trained with L2 regularisation (`λ = 0.001`) on all weight layers.
- Decision threshold: **0.5** (configurable in `backend/main.py`).
- Model weights stored in `cnn_crack_model_final.h5` and tracked with **Git LFS**.

The Grad-CAM overlay targets the `conv2d_1` layer. Crack masks are refined with CLAHE local contrast normalisation, dark-region contour selection, and a morphological open/close pass — making the highlight robust across varying surface lighting.

---

## Feedback & Retraining

When a user flags a wrong verdict, the app:
1. Saves the image to a `feedback/` directory.
2. Saves a JSON sidecar with the timestamp, predicted label, correct label, and raw probability.
3. Returns a confirmation to the UI.

**Feedback does not retrain the deployed model live.** Reports accumulate for periodic offline review → dataset augmentation → full retraining → model versioning. See `backend/main.py` for the feedback endpoint.

---

## Getting Started

### Prerequisites

- Docker (recommended) — or Node 20 + Python 3.11 for manual setup.
- Git LFS — required to pull `cnn_crack_model_final.h5`.

```bash
git lfs install
git clone https://github.com/YOUR_USERNAME/steel-crack-detector.git
cd steel-crack-detector
```

### Run with Docker

```bash
docker build -t crack-detector .
docker run -p 7860:7860 crack-detector
```

Open `http://localhost:7860`.

### Run manually (development)

**Frontend**

```bash
npm install
npm run dev       # Vite dev server on http://localhost:5173
```

**Backend**

```bash
pip install -r requirements.txt
uvicorn backend.main:api --host 0.0.0.0 --port 7860 --reload
```

> The React dev server proxies `/api/*` to the FastAPI backend automatically via Vite's dev config.

---

## API Reference

### `POST /api/predict`

Run crack detection on an uploaded image.

| Field | Type | Description |
|---|---|---|
| `file` | multipart file | JPG or PNG image of a steel surface |

**Response**

```json
{
  "verdict": "CRACK",
  "has_crack": true,
  "confidence": 0.9231,
  "crack_probability": 0.9231,
  "threshold": 0.5,
  "processed_image": "<base64 PNG with crack overlay>"
}
```

`processed_image` is `null` when `has_crack` is `false`.

---

### `POST /api/feedback`

Report an incorrect verdict for offline dataset improvement.

| Field | Type | Description |
|---|---|---|
| `file` | multipart file | The same image that was predicted |
| `correct_label` | string | `"crack"` or `"no_crack"` |
| `predicted_label` | string | The verdict the model returned |
| `crack_probability` | float | Raw model output from the predict call |

**Response**

```json
{
  "status": "feedback_recorded",
  "message": "Feedback saved for offline model improvement.",
  "feedback_id": "20250520T143201123456Z"
}
```

---

### `GET /health`

Returns `{"status": "ok"}` — useful for container health checks.

---

## Project Structure

```
.
├── backend/
│   └── main.py          # FastAPI app — inference, Grad-CAM, feedback endpoints
├── src/
│   ├── main.jsx         # React app — camera, upload, result display, feedback UI
│   └── styles.css       # Frontend styles
├── index.html           # Vite HTML entrypoint
├── app.py               # Hugging Face Space entrypoint shim
├── Dockerfile           # Multi-stage build (Node frontend → Python backend)
├── package.json         # Frontend dependencies
├── requirements.txt     # Python dependencies
├── cnn_crack_model_final.h5   # Trained CNN weights (Git LFS)
└── .gitattributes       # LFS tracking rules for model files
```

---

## Background

This project was built as a final-year AI course submission at **Thapar Institute of Engineering & Technology (TIET)**. The goal was to go beyond a Jupyter notebook demo and ship a real, deployed product — with a proper frontend, a REST API, explainability tooling, and a user feedback loop.

The original model was trained in a Keras notebook on a publicly available steel surface defect dataset. The deployment stack (React + FastAPI + Docker + Hugging Face Spaces) was designed and built on top of that trained artifact.

---

## License

MIT — see [LICENSE](LICENSE).
