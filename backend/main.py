import base64
import json
import threading
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.regularizers import l2


APP_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = APP_DIR / "cnn_crack_model_final.h5"
STATIC_DIR = APP_DIR / "frontend" / "dist"
FEEDBACK_DIR = APP_DIR / "feedback"
IMAGE_SIZE = (128, 128)
THRESHOLD = 0.5

model_lock = threading.Lock()


def build_model() -> Model:
    inputs = Input(shape=(128, 128, 3))
    x = Conv2D(32, (3, 3), activation="relu", kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)

    x = Conv2D(64, (3, 3), activation="relu", kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)

    x = Flatten()(x)
    x = Dense(128, activation="relu", kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.6)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    cnn = Model(inputs, outputs)
    cnn.load_weights(MODEL_PATH)
    return cnn


if not MODEL_PATH.exists():
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")

model = build_model()
FEEDBACK_DIR.mkdir(exist_ok=True)

api = FastAPI(title="Steel Surface Crack Detector")
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_image(file_bytes: bytes) -> Image.Image:
    try:
        return Image.open(BytesIO(file_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc


def preprocess_image(image: Image.Image) -> np.ndarray:
    resized = image.convert("RGB").resize(IMAGE_SIZE)
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def predict_image(image: Image.Image) -> dict[str, Any]:
    image_array = preprocess_image(image)
    with model_lock:
        crack_probability = float(model.predict(image_array, verbose=0)[0][0])

    has_crack = crack_probability >= THRESHOLD
    verdict = "CRACK" if has_crack else "NO CRACK"
    confidence = crack_probability if has_crack else 1.0 - crack_probability

    return {
        "verdict": verdict,
        "has_crack": has_crack,
        "confidence": confidence,
        "crack_probability": crack_probability,
        "threshold": THRESHOLD,
    }


def make_gradcam_heatmap(image_array: np.ndarray) -> np.ndarray:
    last_conv_layer = model.get_layer("conv2d_1")
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)

    if max_value == 0:
        return np.zeros(IMAGE_SIZE, dtype=np.float32)

    return (heatmap / max_value).numpy()


def make_crack_mask(image_rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Cracks in this dataset are usually the dominant dark connected structure.
    # Use a strict darkness threshold first to avoid painting every surface texture.
    dark_cutoff = np.percentile(gray, 24)
    dark_mask = np.uint8(gray <= dark_cutoff) * 255
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)

    candidates = []
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image_rgb.shape[0] * image_rgb.shape[1]
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if min(w, h) == 0 or area < max(30, image_area * 0.0005):
            continue
        elongation = max(w, h) / min(w, h)
        extent = area / float(w * h)
        score = area * (1.0 + min(elongation, 8.0)) * (1.15 - min(extent, 0.9))
        candidates.append((score, contour))

    selected = np.zeros_like(dark_mask)
    if candidates:
        _, best_contour = max(candidates, key=lambda item: item[0])
        cv2.drawContours(selected, [best_contour], -1, 255, thickness=cv2.FILLED)
    else:
        selected = dark_mask

    selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    selected = cv2.dilate(selected, np.ones((5, 5), np.uint8), iterations=1)

    # Grad-CAM is used only as a loose sanity constraint. If it cuts away too much,
    # keep the dominant dark structure rather than adding noisy heatmap islands.
    heatmap_resized = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]))
    focus_cutoff = np.percentile(heatmap_resized, 45)
    focus_mask = np.uint8(heatmap_resized >= focus_cutoff) * 255
    focus_mask = cv2.dilate(focus_mask, np.ones((17, 17), np.uint8), iterations=1)

    constrained = cv2.bitwise_and(selected, focus_mask)
    final_mask = constrained if cv2.countNonZero(constrained) > max(25, cv2.countNonZero(selected) * 0.35) else selected
    return final_mask


def heatmap_overlay_base64(image: Image.Image) -> str:
    image_array = preprocess_image(image)
    with model_lock:
        heatmap = make_gradcam_heatmap(image_array)

    image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask = make_crack_mask(image_rgb, heatmap)

    overlay = image_rgb.copy()
    green_layer = np.zeros_like(image_rgb)
    green_layer[:, :] = [42, 145, 92]
    overlay = cv2.addWeighted(overlay, 0.86, green_layer, 0.14, 0)

    red_layer = np.zeros_like(image_rgb)
    red_layer[:, :] = [220, 30, 30]
    crack_overlay = cv2.addWeighted(image_rgb, 0.32, red_layer, 0.68, 0)
    overlay[mask > 0] = crack_overlay[mask > 0]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 28, 28), thickness=2)

    output = Image.fromarray(overlay)
    buffer = BytesIO()
    output.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@api.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@api.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    file_bytes = await file.read()
    image = read_image(file_bytes)
    result = predict_image(image)
    result["processed_image"] = heatmap_overlay_base64(image) if result["has_crack"] else None
    return result


@api.post("/api/feedback")
async def feedback(
    file: UploadFile = File(...),
    correct_label: str = Form(...),
    predicted_label: str = Form(""),
    crack_probability: float = Form(0.0),
) -> dict[str, Any]:
    if correct_label not in {"crack", "no_crack"}:
        raise HTTPException(status_code=400, detail="correct_label must be crack or no_crack")

    file_bytes = await file.read()
    image = read_image(file_bytes)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    image_path = FEEDBACK_DIR / f"{timestamp}.jpg"
    metadata_path = FEEDBACK_DIR / f"{timestamp}.json"

    image.save(image_path, quality=92)
    metadata = {
        "timestamp": timestamp,
        "correct_label": correct_label,
        "predicted_label": predicted_label,
        "crack_probability": crack_probability,
        "source_filename": file.filename,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "status": "feedback_recorded",
        "message": "Feedback saved for offline model improvement.",
        "feedback_id": timestamp,
    }


if STATIC_DIR.exists():
    api.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")


@api.get("/{path:path}")
def serve_frontend(path: str) -> FileResponse:
    requested_path = STATIC_DIR / path
    if path and requested_path.exists() and requested_path.is_file():
        return FileResponse(requested_path)
    return FileResponse(STATIC_DIR / "index.html")
