---
title: Steel Surface Crack Detector
colorFrom: orange
colorTo: gray
sdk: docker
pinned: false
app_file: app.py
---

# Steel Surface Crack Detector

React + FastAPI Hugging Face Space for steel surface crack detection.

Features:

- Browser camera capture for live steel surface photos.
- Upload fallback for local test images.
- CNN crack/no-crack prediction with confidence score.
- Red crack-region overlay returned by the backend only when the model predicts CRACK.
- User correction flow that stores feedback images.
- Report-only feedback confirmation for incorrect verdicts.

Important note: feedback does not retrain the deployed model live. Reports are saved for
offline review, dataset improvement, retraining, validation, and future model versioning.
