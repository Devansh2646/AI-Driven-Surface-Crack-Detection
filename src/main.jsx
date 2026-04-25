import React, { useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Camera,
  CheckCircle,
  FileArrowUp,
  Flag,
  ShieldCheck,
  Warning,
  X,
} from '@phosphor-icons/react';
import './styles.css';

const verdictToCorrection = {
  CRACK: 'no_crack',
  'NO CRACK': 'crack',
};

const correctionCopy = {
  crack: 'CRACK',
  no_crack: 'NO CRACK',
};

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [feedback, setFeedback] = useState(null);
  const [loading, setLoading] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [error, setError] = useState('');

  async function startCamera() {
    setError('');
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
        audio: false,
      });
      videoRef.current.srcObject = mediaStream;
      setStream(mediaStream);
      setCameraReady(true);
    } catch {
      setError('Camera access failed. Allow browser permission or use image upload.');
    }
  }

  function stopCamera() {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }
    setStream(null);
    setCameraReady(false);
  }

  function resetResult(file, objectUrl) {
    setImageFile(file);
    setPreviewUrl(objectUrl);
    setPrediction(null);
    setFeedback(null);
    setConfirmOpen(false);
  }

  async function capturePhoto() {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);

    canvas.toBlob(async (blob) => {
      const file = new File([blob], 'camera-steel-surface.jpg', { type: 'image/jpeg' });
      resetResult(file, URL.createObjectURL(file));
      await predict(file);
    }, 'image/jpeg', 0.92);
  }

  async function handleUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    resetResult(file, URL.createObjectURL(file));
    await predict(file);
  }

  async function predict(file) {
    setLoading(true);
    setError('');
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Prediction request failed');
      setPrediction(await response.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function reportCorrection() {
    if (!imageFile || !prediction) return;

    const correctLabel = verdictToCorrection[prediction.verdict];
    setLoading(true);
    setError('');
    try {
      const formData = new FormData();
      formData.append('file', imageFile);
      formData.append('correct_label', correctLabel);
      formData.append('predicted_label', prediction.verdict);
      formData.append('crack_probability', prediction.crack_probability);

      const response = await fetch('/api/feedback', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Feedback request failed');
      setFeedback(await response.json());
      setConfirmOpen(false);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const correction = prediction ? verdictToCorrection[prediction.verdict] : null;

  return (
    <main className="shell">
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Surface inspection CNN</p>
          <h1>Steel crack review</h1>
          <p className="lede">
            Capture or upload a steel surface image. The trained CNN returns a strict
            CRACK or NO CRACK verdict, confidence score, a red crack highlight, and
            a subtle green wash for non-crack surface area.
          </p>
        </div>
        <div className="hero-steps" aria-label="How to use this demo">
          <span>01 Capture or upload</span>
          <span>02 Review original and processed output</span>
          <span>03 Report incorrect verdicts for offline retraining</span>
        </div>
      </section>

      <section className="input-grid">
        <div className="panel camera-panel">
          <div className="panel-title">
            <Camera size={22} weight="duotone" />
            Camera capture
          </div>
          <div className="camera-frame">
            <video ref={videoRef} autoPlay playsInline muted />
            {!cameraReady && <div className="camera-placeholder">Camera preview</div>}
          </div>
          <canvas ref={canvasRef} hidden />
          <div className="actions">
            {!cameraReady ? (
              <button onClick={startCamera}>Start camera</button>
            ) : (
              <>
                <button onClick={capturePhoto} disabled={loading}>Capture and process</button>
                <button className="ghost" onClick={stopCamera}>Stop camera</button>
              </>
            )}
          </div>
        </div>

        <div className="panel upload-panel">
          <div className="panel-title">
            <FileArrowUp size={22} weight="duotone" />
            Upload image
          </div>
          <label className="upload-box">
            <input type="file" accept="image/png,image/jpeg,image/jpg" onChange={handleUpload} />
            <span>Choose a steel surface image</span>
            <small>JPG, JPEG, or PNG</small>
          </label>
        </div>
      </section>

      {error && <div className="error">{error}</div>}
      {loading && <div className="loading">Processing image and running inference...</div>}

      {prediction && (
        <section className="result-card">
          <div className="result-topline">
            <div>
              <p className="eyebrow">Model verdict</p>
              <div className="verdict-row">
                <h2 className={prediction.has_crack ? 'verdict danger-text' : 'verdict safe-text'}>
                  {prediction.verdict}
                </h2>
                <button
                  className="report-icon"
                  onClick={() => setConfirmOpen(true)}
                  aria-label="Report wrong prediction"
                  title="Report wrong prediction"
                >
                  <Flag size={20} weight="duotone" />
                </button>
              </div>
            </div>
            <div className="confidence-block">
              <span>Confidence</span>
              <strong>{(prediction.confidence * 100).toFixed(2)}%</strong>
            </div>
          </div>

          <div className="meter">
            <span style={{ width: `${prediction.confidence * 100}%` }} />
          </div>

          <div className="image-compare">
            <figure>
              <figcaption>Original image</figcaption>
              <img src={previewUrl} alt="Original steel surface input" />
            </figure>

            <figure>
              <figcaption>{prediction.has_crack ? 'Processed image' : 'Processed result'}</figcaption>
              {prediction.has_crack && prediction.processed_image ? (
                <img
                  src={`data:image/png;base64,${prediction.processed_image}`}
                  alt="Processed crack highlight"
                />
              ) : (
                <div className="no-crack-panel">
                  <ShieldCheck size={54} weight="duotone" />
                  <strong>NO CRACK</strong>
                  <span>No crack-highlight overlay generated for a clean verdict.</span>
                </div>
              )}
            </figure>
          </div>

          <div className="raw-score">
            Raw crack probability: {prediction.crack_probability.toFixed(4)} | Decision threshold:
            {` ${prediction.threshold}`}
          </div>

          {feedback && (
            <div className="feedback-success">
              <CheckCircle size={22} weight="duotone" />
              <div>
                <strong>Feedback recorded</strong>
                <span>{feedback.message}</span>
              </div>
            </div>
          )}
        </section>
      )}

      {confirmOpen && prediction && (
        <div className="modal-backdrop" role="presentation">
          <div className="modal" role="dialog" aria-modal="true" aria-labelledby="confirm-title">
            <button className="modal-close" onClick={() => setConfirmOpen(false)} aria-label="Close">
              <X size={18} />
            </button>
            <Warning size={34} weight="duotone" className="danger-text" />
            <h3 id="confirm-title">Confirm wrong verdict report</h3>
            <p>
              The model predicted <strong>{prediction.verdict}</strong>. Confirm that the correct
              verdict should be <strong>{correctionCopy[correction]}</strong>.
            </p>
            <p className="modal-note">
              This stores the image and correction for offline dataset improvement. It does not
              retrain the deployed model during this session.
            </p>
            <div className="modal-actions">
              <button className="ghost" onClick={() => setConfirmOpen(false)}>Cancel</button>
              <button onClick={reportCorrection} disabled={loading}>Confirm report</button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
