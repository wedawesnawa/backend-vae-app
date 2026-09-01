# VAE Audio Reconstruction Backend

Backend API for **audio preprocessing, log-Mel spectrogram extraction, VAE reconstruction, audio reconstruction**, and reconstruction metric calculation using **Python, Flask, and PyTorch**.

## Tech Stack

* **Python**
* **Flask**: REST API framework
* **PyTorch**: Variational Autoencoder (VAE)
* **Librosa**: Audio processing and feature extraction
* **SoundFile**: Audio file I/O
* **NumPy**: Numerical computation

## Features

* Audio preprocessing
* Log-Mel spectrogram extraction
* Spectrogram normalization
* VAE-based audio reconstruction
* Mel spectrogram to audio reconstruction
* Reconstruction metric calculation
* Latent representation extraction (`mu` and `logvar`)
* Single audio processing
* Batch audio processing
* Health check endpoint
* Model information endpoint
* Generated audio output

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/wedawesnawa/backend-vae-app.git
cd backend-vae-app
```

## 2. Create a Virtual Environment

A virtual environment is recommended to isolate the project's dependencies from the system Python environment.

### Windows

```bash
python -m venv venv
```

## 3. Activate the Virtual Environment

### Windows — Command Prompt

```bash
venv\Scripts\activate
```

### Windows — PowerShell

```powershell
venv\Scripts\Activate.ps1
```

If the virtual environment is activated successfully, `(venv)` will appear at the beginning of your terminal prompt.

Example:

```text
(venv) C:\project\backend-vae-app>
```

## 4. Install Dependencies

Make sure the virtual environment is active, then install the required Python packages:

```bash
pip install -r requirements.txt
```

> Make sure the dependency file is named `requirements.txt`. If your repository uses `requiredment.txt`, use `pip install -r requiredment.txt` instead.

## 5. Run the Backend

After all dependencies have been installed, run:

```bash
python app.py
```

The backend will be available at:

```text
http://localhost:5000
```

The Flask application runs using:

```python
app.run(host='0.0.0.0', port=5000, debug=True)
```

---

# API Endpoints

Base URL:

```text
http://localhost:5000
```

## 1. Health Check

### `GET /health`

Checks the backend status, PyTorch device, model loading status, and model configuration.

### Request

```http
GET /health
```

### Response

```json
{
  "status": "healthy",
  "device": "cuda",
  "model_loaded": true,
  "config": {}
}
```

---

## 2. Model Information

### `GET /info`

Returns information about the VAE model, audio configuration, spectrogram parameters, and normalization settings.

### Request

```http
GET /info
```

### Response

```json
{
  "model": {
    "type": "VAE",
    "latent_dim": 128,
    "input_shape": [1, 128, 173]
  },
  "audio_params": {
    "sample_rate": 22050,
    "duration": 5,
    "samples": 110250
  },
  "spectrogram_params": {
    "n_fft": 2048,
    "hop_length": 512,
    "n_mels": 128,
    "fmax": 11025
  },
  "normalization": {
    "global_min": 0,
    "global_max": 1
  }
}
```

> The values returned by this endpoint depend on the model configuration used by the backend.

---

# 3. Process Single Audio

### `POST /process`

Processes a single WAV audio file through the complete VAE reconstruction pipeline.

### Content-Type

```text
multipart/form-data
```

### Form Data

| Parameter         | Type  | Required | Description                                    |
| ----------------- | ----- | -------- | ---------------------------------------------- |
| `file`            | File  | Yes      | WAV audio file                                 |
| `noise_threshold` | Float | No       | Noise threshold used during VAE reconstruction |

### Example

```bash
curl -X POST http://localhost:5000/process \
  -F "file=@audio.wav" \
  -F "noise_threshold=0.20"
```

### Processing Pipeline

```text
WAV Audio
    ↓
Audio Preprocessing
    ↓
Log-Mel Spectrogram Extraction
    ↓
Normalization [0, 1]
    ↓
VAE Reconstruction
    ↓
Denormalization
    ↓
Mel Spectrogram → Audio
    ↓
Metrics Calculation
    ↓
JSON Response
```

### Response

```json
{
  "success": true,
  "filename": "audio.wav",
  "preprocessing": {},
  "feature_extraction": {},
  "spectrograms": {
    "original": "...",
    "reconstructed": "..."
  },
  "audio": {
    "original": "...",
    "reconstructed": "..."
  },
  "metrics": {},
  "latent_representation": {
    "mu": [],
    "logvar": []
  }
}
```

### Response Components

#### `preprocessing`

Contains information about the audio preprocessing steps.

```json
{
  "steps": [],
  "summary": {
    "duration": "5s",
    "sample_rate": 22050,
    "samples": 110250
  }
}
```

#### `feature_extraction`

Contains information about the extracted log-Mel spectrogram.

```json
{
  "steps": [],
  "summary": {
    "n_mels": 128,
    "time_frames": 173,
    "spectrogram_shape": [128, 173]
  }
}
```

#### `spectrograms`

Contains the original and reconstructed spectrograms encoded as Base64.

```json
{
  "original": "...",
  "reconstructed": "..."
}
```

#### `audio`

Contains the original and reconstructed audio encoded as Base64.

```json
{
  "original": "...",
  "reconstructed": "..."
}
```

#### `metrics`

Contains the evaluation metrics calculated between the original and reconstructed spectrograms.

The metric structure is generated by:

```python
calculate_metrics(...)
```

#### `latent_representation`

Contains the VAE latent representation:

```json
{
  "mu": [],
  "logvar": []
}
```

---

# 4. Process Multiple Audio Files

### `POST /process_batch`

Processes multiple WAV audio files in a single request.

### Content-Type

```text
multipart/form-data
```

### Form Data

| Parameter         | Type  | Required | Description                        |
| ----------------- | ----- | -------- | ---------------------------------- |
| `files[]`         | File  | Yes      | Multiple WAV audio files           |
| `noise_threshold` | Float | No       | Noise threshold, default is `0.20` |

### Example

```bash
curl -X POST http://localhost:5000/process_batch \
  -F "files[]=@audio1.wav" \
  -F "files[]=@audio2.wav" \
  -F "files[]=@audio3.wav" \
  -F "noise_threshold=0.20"
```

### Response

```json
{
  "success": true,
  "total_files": 3,
  "successful": 3,
  "failed": 0,
  "results": [
    {
      "filename": "audio1.wav",
      "success": true,
      "metrics": {},
      "audio_reconstructed": "..."
    },
    {
      "filename": "audio2.wav",
      "success": true,
      "metrics": {},
      "audio_reconstructed": "..."
    }
  ]
}
```

If a file fails during processing, it will still be included in the `results` array:

```json
{
  "filename": "audio.wav",
  "success": false,
  "error": "Error message"
}
```

---


# API Summary

| Method | Endpoint               | Description                       |
| ------ | ---------------------- | --------------------------------- |
| `GET`  | `/health`              | Check backend and model status    |
| `GET`  | `/info`                | Get model and audio configuration |
| `POST` | `/process`             | Process a single WAV file         |
| `POST` | `/process_batch`       | Process multiple WAV files        |

---

# Project Structure

```text
backend/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   └── ...
│
├── temp_uploads/
│   └── ...
│
├── temp_outputs/
│   └── ...
│
└── venv/
    └── ...
```

> The actual folder and file names may vary depending on the project structure.

---

# Environment

The backend uses the following default configuration:

```text
Host  : 0.0.0.0
Port  : 5000
Debug : True
```

To access the backend locally:

```text
http://localhost:5000
```

To check the backend status:

```text
http://localhost:5000/health
```

---

# Supported Audio Format

The `/process` and `/process_batch` endpoints currently support:

```text
.wav
```

Files with other formats will be rejected.

---

# License

This project is developed for research and educational purposes.
