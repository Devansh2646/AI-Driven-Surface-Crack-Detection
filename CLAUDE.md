# CLAUDE.md

## Current Deployment Goal

This folder contains the Hugging Face Space deployment for a steel surface crack detection demo.
The evaluator should be able to open the Space, read the instructions, capture or upload a steel
surface image, and receive a clear model result.

## Current Architecture

- `Dockerfile`: Docker Space runtime. Builds the React frontend, installs the Python backend, and serves FastAPI on port `7860`.
- `app.py`: Compatibility FastAPI/uvicorn entrypoint for environments that expect a root Python app file.
- `backend/main.py`: FastAPI + TensorFlow inference backend.
- `src/main.jsx`: React UI for camera capture, upload, result display, and feedback reporting.
- `src/styles.css`: Custom responsive frontend styling.
- `cnn_crack_model_final.h5`: Saved CNN model artifact. Track with Git LFS.
- `README.md`: Hugging Face Space metadata using `sdk: docker`.

## User Flow

1. User sees a hero/instruction page explaining the demo.
2. User captures a photo or uploads an image.
3. Backend returns a strict verdict: `CRACK` or `NO CRACK`.
4. Frontend shows the original image and processed result.
5. On desktop, original and processed images appear side by side.
6. On mobile, original appears first and processed result appears below.
7. If verdict is `CRACK`, backend returns a processed image with a red crack-region overlay.
8. If verdict is `NO CRACK`, frontend shows a clean `NO CRACK` processed panel instead of a heatmap.
9. A report icon beside the verdict lets the user report a wrong prediction.
10. Reporting opens a confirmation modal.
11. Confirmed feedback saves the image and metadata for offline dataset improvement.

## Important Product Decision

The app must not retrain the model live and must not show a retraining/weight animation.

Reason: one-image online training from unverified user feedback is not a valid production ML loop
and may degrade the model. For this class demo, the technically honest flow is feedback collection
for offline review and future retraining.

## Backend API

`POST /api/predict`

Returns:

```json
{
  "verdict": "CRACK",
  "has_crack": true,
  "confidence": 0.94,
  "crack_probability": 0.94,
  "threshold": 0.5,
  "processed_image": "base64-red-overlay-or-null"
}
```

`POST /api/feedback`

Saves the image and metadata in `feedback/` and returns:

```json
{
  "status": "feedback_recorded",
  "message": "Feedback saved for offline model improvement.",
  "feedback_id": "timestamp"
}
```

## Verification

The following checks passed after the latest changes:

```bash
python -m py_compile backend/main.py
npm run build
```

## Deployment

From inside `app/`:

```bash
git init
git lfs install
git add .
git commit -m "Deploy steel crack detector"
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git push -u origin main
```

## Future Notes

- Do not commit `node_modules/`, `dist/`, `feedback/`, or Python cache folders.
- Do not deploy the training dataset or notebook.
- Keep the TensorFlow architecture in `backend/main.py` aligned with the saved `.h5` file.
- Feedback examples should be reviewed offline before retraining any model.
