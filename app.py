# app.py
import numpy as np
import librosa
import scipy.signal as signal
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import noisereduce as nr
import scipy.signal as signal
import pyloudnorm as pyln
import traceback
import io
import json
import base64
import os
import scipy.signal
import warnings

# ==================== KONFIGURASI ====================
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'temp_uploads'
OUTPUT_FOLDER = 'temp_outputs'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Parameter audio — harus IDENTIK dengan training
TARGET_SR = 22050
TARGET_DURATION = 4
TARGET_SAMPLES = TARGET_SR * TARGET_DURATION  # 88200

# Parameter spektrogram — harus IDENTIK dengan training
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 0
FMAX = 11025
TOP_DB = 80
PREEMPHASIS_COEF = 0.97
GLOBAL_MIN = -80.0
GLOBAL_MAX = 0.0
TARGET_LUFS = -20.0

# BUG FIX #1: nama file model disesuaikan dengan yang disimpan di notebook
# Notebook menyimpan: 'vae_gamelan_final_1.pth', bukan 'vae_gamelan_final_2.pth'
MODEL_PATH = 'models/vae_gamelan_final_lord_try.pth'
CONFIG_PATH = 'models/model_config.json'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==================== DEFINISI MODEL VAE ====================
# Arsitektur harus IDENTIK dengan yang dipakai saat training
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder: Input (1, 128, 173) -> (128, 16, 22)
        self.enc1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.enc3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.flatten_dim = 128 * 16 * 22

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)

        # Decoder: Upsample + Conv (sama persis dengan notebook)
        self.up1 = nn.Upsample(size=(32, 44), mode='nearest')
        self.dec1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.up2 = nn.Upsample(size=(64, 87), mode='nearest')
        self.dec2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        self.up3 = nn.Upsample(size=(128, 173), mode='nearest')
        self.dec3 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        x = F.relu(self.bn1(self.enc1(x)))
        x = F.relu(self.bn2(self.enc2(x)))
        x = F.relu(self.bn3(self.enc3(x)))

        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)

        mu = self.fc_mu(x_flat)
        logvar = self.fc_logvar(x_flat)
        z = self.reparameterize(mu, logvar)

        x = self.decoder_input(z)
        x = x.view(batch_size, 128, 16, 22)

        x = self.up1(x)
        x = self.dec1(x)
        x = F.relu(self.bn4(x))

        x = self.up2(x)
        x = self.dec2(x)
        x = F.relu(self.bn5(x))

        x = self.up3(x)
        reconstruction = torch.sigmoid(self.dec3(x))

        return reconstruction, mu, logvar

# ==================== LOAD MODEL ====================
def load_model_and_config():
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    model = VAE(latent_dim=config['latent_dim']).to(device)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model loaded successfully")
    else:
        print(f"WARNING: Model file not found at {MODEL_PATH}")

    model.eval()
    return model, config

model, model_config = load_model_and_config()

# ==================== PREPROCESSING ====================
# def estimate_noise_floor(audio, sr):
#     rms_frames = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
#     n_quiet = max(1, int(len(rms_frames) * 0.10))
#     quiet_rms = np.sort(rms_frames)[:n_quiet].mean()
#     return 20 * np.log10(quiet_rms + 1e-10)

# def smart_denoise(audio, sr=22050):
#     floor_db = estimate_noise_floor(audio, sr)
#     if floor_db < -58:
#         return audio
#     prop = 0.15 if floor_db < -50 else 0.40
#     noise_sample = audio[:int(sr * 0.2)]
#     return nr.reduce_noise(
#         y=audio, sr=sr,
#         y_noise=noise_sample,
#         prop_decrease=prop,
#         stationary=True,
#         freq_mask_smooth_hz=500
#     )

def pad_with_fadeout(audio, target_len):
    if len(audio) < target_len:
        fade_len = max(1, int(len(audio) * 0.10))
        fade_curve = np.linspace(1.0, 0.0, fade_len)
        audio = audio.copy()
        audio[-fade_len:] *= fade_curve
        audio = np.pad(audio, (0, target_len - len(audio)))
    return audio

def safe_normalize(audio, target_db=-3.0):
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        current_db = 20 * np.log10(max_val)
        if current_db > -40:
            target_amp = 10 ** (target_db / 20)
            audio = audio * (target_amp / max_val)
    return audio

def lufs_normalize(audio, sr, target_lufs=TARGET_LUFS):
    try:
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio)
        if not np.isfinite(loudness):
            return audio
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio_normalized = pyln.normalize.loudness(audio, loudness, target_lufs)
        return audio_normalized.astype(np.float32)
    except Exception as e:
        print(f"  [LUFS] Gagal: {e}, skip.")
        return audio

def preprocess_audio(file_path):
    preprocessing_steps = []
    # 1. Load & Resample
    y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
    preprocessing_steps.append({
        'step': 'Loading',
        'description': f'Audio loaded: {len(y)/TARGET_SR:.2f}s, SR: {sr}Hz'
    })
    # 2. Smart Denoise
    # y = smart_denoise(y, TARGET_SR)
    # preprocessing_steps.append({
    #     'step': 'Denoising',
    #     'description': 'Applied smart denoise based on noise floor'
    # })

    # max_val = np.max(np.abs(y))
    # if max_val > 0:
    #     y = y / max_val
    # preprocessing_steps.append({
    #     'step': 'Normalization',
    #     'description': f'Amplitude normalized to [-1, 1], max was: {max_val:.4f}'
    # })

    # original_length = len(y)
    # if len(y) > TARGET_SAMPLES:
    #     y = y[:TARGET_SAMPLES]
    #     action = 'truncated'
    # elif len(y) < TARGET_SAMPLES:
    #     padding = TARGET_SAMPLES - len(y)
    #     y = np.pad(y, (0, padding), mode='constant')
    #     action = 'padded'
    # else:
    #     action = 'exact'

    # preprocessing_steps.append({
    #     'step': 'Duration adjustment',
    #     'description': f'Audio {action} to {TARGET_DURATION}s ({TARGET_SAMPLES} samples)',
    #     'original_length': original_length,
    #     'final_length': len(y)
    # })

    # 3. Truncate atau Pad
    original_length = len(y)
    if len(y) > TARGET_SAMPLES:
        y = y[:TARGET_SAMPLES]
        action = 'truncated'
    else:
        y = pad_with_fadeout(y, TARGET_SAMPLES)
        action = 'padded with fadeout'

    preprocessing_steps.append({
        'step': 'Duration adjustment',
        'description': f'Audio {action} to {TARGET_DURATION}s ({TARGET_SAMPLES} samples)',
        'original_length': original_length,
        'final_length': len(y)
    })

    # 4. Normalisasi
    y = safe_normalize(y, target_db=-3.0)
    preprocessing_steps.append({
        'step': 'Normalization',
        'description': 'Safe normalized to -3.0 dB'
    })

    # 5. LUFS
    y = lufs_normalize(y, TARGET_SR, target_lufs=TARGET_LUFS)
    preprocessing_steps.append({'step': 'LUFS Normalization', 'description': f'Loudness normalized to {TARGET_LUFS} LUFS'})

    return y, preprocessing_steps


def extract_log_mel(y):
    extraction_steps = []
    # 1. Pre-emphasis
    # y_preemph = librosa.effects.preemphasis(y, coef=PREEMPHASIS_COEF)
    # extraction_steps.append({
    #     'step': 'Pre-emphasis',
    #     'description': f'Applied pre-emphasis (coef={PREEMPHASIS_COEF})'
    # })

    # 2. Hitung Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX
    )

    # mel_spec = librosa.feature.melspectrogram(
    #     y=y_preemph,
    #     sr=TARGET_SR,
    #     n_fft=N_FFT,
    #     hop_length=HOP_LENGTH,
    #     n_mels=N_MELS,
    #     fmax=TARGET_SR / 2  # Nyquist, sama dengan notebook
    # )
    extraction_steps.append({
        'step': 'Mel spectrogram',
        'description': f'Shape: {mel_spec.shape}'
    })

    # log_mel = librosa.power_to_db(mel_spec, ref=1.0)
    # extraction_steps.append({
    #     'step': 'Log conversion (power_to_db)',
    #     'description': f'Range: {log_mel.min():.1f} ~ {log_mel.max():.1f} dB'
    # })

    # 3. Konversi ke Log Scale (Power to DB) - SESUAI NOTEBOOK
    # ref_val = float(mel_spec.max())
    log_mel = librosa.power_to_db(mel_spec, ref=1.0, top_db=TOP_DB)
    extraction_steps.append({'step': 'Log conversion', 'description': f'Range: {log_mel.min():.1f} ~ {log_mel.max():.1f} dB (ref=1.0)'})

    return log_mel.astype(np.float32), extraction_steps


def normalize_spectrogram(spec):
    spec_norm = (spec - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN)
    spec_norm = np.clip(spec_norm, 0, 1)
    return spec_norm.astype(np.float32)


def denormalize_spectrogram(spec_norm):
    spec_db = spec_norm * (GLOBAL_MAX - GLOBAL_MIN) + GLOBAL_MIN
    return spec_db


# ==================== REKONSTRUKSI VAE ====================
def reconstruct_vae(spec_norm, noise_threshold=0.15):
    spec_tensor = torch.from_numpy(spec_norm).float().to(device)
    spec_tensor = spec_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 128, 173)

    with torch.no_grad():
        recon_tensor, mu, logvar = model(spec_tensor)

    recon_spec_norm = recon_tensor.squeeze().cpu().numpy()
    recon_spec_norm = np.clip(recon_spec_norm, 0, 1)

    # Terapkan Noise Gate (Thresholding)
    if noise_threshold > 0.0:
        recon_spec_norm = np.where(recon_spec_norm < noise_threshold, 0.0, recon_spec_norm)

    return recon_spec_norm, mu, logvar

# ==================== KONVERSI KE AUDIO ====================
# def mel_to_audio(mel_spec_db, sr=22050, n_fft=2048, hop_length=512,
#                  preemphasis_coef=0.97, n_iter=64, apply_dsp=False):
#     try:
#         mel_spec_power = librosa.db_to_power(mel_spec_db, ref=1.0)

#         audio_signal = librosa.feature.inverse.mel_to_audio(
#             mel_spec_power,
#             sr=sr,
#             n_fft=n_fft,
#             hop_length=hop_length,
#             n_iter=128,
#             fmax=sr / 2
#         )
#         audio_signal = librosa.effects.deemphasis(audio_signal, coef=preemphasis_coef)
#         max_val = np.max(np.abs(audio_signal))
#         if max_val > 0:
#             audio_signal = audio_signal / max_val

#         return audio_signal

#     except Exception as e:
#         print(f"Error in mel_to_audio: {e}")
#         traceback.print_exc()
def mel_to_audio(mel_spec_db, sr=22050, n_fft=2048, hop_length=512, preemphasis_coef=0.97):
    """Sesuai fungsi inverse_normalize_and_convert di notebook"""
    try:
        # Konversi dB ke Power
        mel_spec_power = librosa.db_to_power(mel_spec_db)

        # Inversi Mel -> Audio via Griffin-Lim
        audio_signal = librosa.feature.inverse.mel_to_audio(
            mel_spec_power,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length
        )

        # # DE-EMPHASIS
        # audio_signal = librosa.effects.deemphasis(audio_signal, coef=preemphasis_coef)

        # Normalisasi ulang amplitude
        max_val = np.max(np.abs(audio_signal))
        if max_val > 0:
            audio_signal = audio_signal / max_val

        return audio_signal

    except Exception as e:
        print(f"Error in mel_to_audio: {e}")
        traceback.print_exc()


def mel_to_audio_with_feedback(mel_spec_db, y_original=None, sr=22050, n_fft=2048, hop_length=512):
    try:
        # 1. Konversi dB ke Power
        mel_spec_power = librosa.db_to_power(mel_spec_db)

        # 2. Perkiraan Linear STFT Magnitude (mag_original) dari Mel-Spectrogram
        mag_approx = librosa.feature.inverse.mel_to_stft(
            mel_spec_power, sr=sr, n_fft=n_fft, power=1.0
        )

        # 3. Base Reconstruction menggunakan Griffin-Lim biasa
        y_current = librosa.griffinlim(mag_approx, n_iter=64, hop_length=hop_length)

        # 4. Iterative Feedback Loop (Sesuai kodemu)
        if y_original is not None:
            n_refinement = 5
            print("Memulai Refinement Berbasis Feedback...")
            
            for i in range(n_refinement):
                min_len = min(len(y_original), len(y_current))
                
                # Hitung residual
                res = y_original[:min_len] - y_current[:min_len]

                # Reinjection
                y_current[:min_len] = y_current[:min_len] + 0.5 * res

                # Project back to STFT space
                D_refine = librosa.stft(y_current, n_fft=n_fft, hop_length=hop_length)
                _, phase_refine = librosa.magphase(D_refine)

                # Rekonstruksi ulang menggunakan mag_approx dan phase_refine
                y_current = librosa.istft(mag_approx * phase_refine, hop_length=hop_length)

                # current_rmse = np.sqrt(np.mean((y_original[:min_len] - y_current[:min_len])**2))
                # print(f"Iterasi Feedback {i+1}: RMSE = {current_rmse:.5f}")

        # 5. DE-EMPHASIS (Wajib agar tidak cempreng)
        # audio_signal = librosa.effects.deemphasis(y_current, coef=preemphasis_coef)
        audio_signal = y_current

        # 6. Normalisasi ulang amplitude
        max_val = np.max(np.abs(audio_signal))
        if max_val > 0:
            audio_signal = audio_signal / max_val

        return audio_signal

    except Exception as e:
        print(f"Error in mel_to_audio_with_feedback: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==================== METRIK ====================
def calculate_metrics(original_spec_norm, recon_spec_norm, log_mel_spec, recon_spec_db, mu, logvar):
    metrics = {}
 
    # 1. Hitung MAE dan MSE pada [0,1]
    original_norm = np.clip(original_spec_norm, 0, 1)
    recon_norm    = np.clip(recon_spec_norm,    0, 1)
    
    # Tambahkan perhitungan MAE (L1) menggunakan numpy (karena datanya array numpy)
    metrics['mae'] = float(np.mean(np.abs(original_norm - recon_norm)))
    # metrics['mse'] = float(np.mean((original_norm - recon_norm) ** 2))
    mse_val = float(np.mean((original_norm - recon_norm) ** 2))
    # metrics['mse'] = float(np.sqrt(mse_val))
    metrics['mse'] = mse_val
    # 2. KLD dari latent space
    if torch.is_tensor(mu):
        mu_cpu     = mu.cpu()
        logvar_cpu = logvar.cpu()
    else:
        mu_cpu, logvar_cpu = mu, logvar
 
    kld = -0.5 * torch.sum(1 + logvar_cpu - mu_cpu.pow(2) - logvar_cpu.exp())
    metrics['kld']  = (float(kld) / mu_cpu.size(0)) * 0.00025
    
    # ELBO sekarang menggunakan MAE, konsisten dengan pelatihan
    metrics['elbo'] = metrics['mae'] + metrics['kld']
 
    # 3. LSD di domain mel dB — SAMA PERSIS dengan calculate_lsd() notebook:
    #    orig_db  : hasil denormalisasi [0,1] -> dB  (range [-80, 0])
    #    recon_db : hasil denormalisasi output VAE -> dB
    #    Rumus  : sqrt(mean((orig_db - recon_db)^2, axis=mel_bins)) per time frame
    diff_sq      = (log_mel_spec - recon_spec_db) ** 2          # (128, T)
    lsd_per_frame = np.sqrt(np.mean(diff_sq, axis=0))            # (T,)
    metrics['lsd'] = {
        'mean': float(np.mean(lsd_per_frame)),
        'std':  float(np.std(lsd_per_frame)),
        'unit': 'dB',
        'note': 'Dihitung di domain mel spectrogram (dB), konsisten dengan notebook (train~2.54dB, test~2.84dB)'
    }
 
    return metrics

# def calculate_metrics(original_spec_norm, recon_spec_norm, log_mel_spec, recon_spec_db, mu, logvar):
#     metrics = {}
 
#     original_norm = np.clip(original_spec_norm, 0, 1)
#     recon_norm    = np.clip(recon_spec_norm,    0, 1)
    
#     metrics['mae'] = float(np.mean(np.abs(original_norm - recon_norm)))
#     metrics['mse'] = float(np.mean((original_norm - recon_norm) ** 2))
 
#     if torch.is_tensor(mu):
#         mu_cpu     = mu.cpu()
#         logvar_cpu = logvar.cpu()
#     else:
#         mu_cpu, logvar_cpu = mu, logvar
 
#     kld = -0.5 * torch.sum(1 + logvar_cpu - mu_cpu.pow(2) - logvar_cpu.exp())
#     metrics['kld']  = (float(kld) / mu_cpu.size(0)) * 0.00025
#     metrics['elbo'] = metrics['mae'] + metrics['kld']
 
#     # LSD di domain mel dB (mengikuti evaluate_metrics_audio di notebook)
#     diff_sq = (log_mel_spec - recon_spec_db) ** 2          # (128, 173)
#     lsd_val = np.sqrt(np.mean(diff_sq)) # RMSE keseluruhan
    
#     metrics['lsd'] = {
#         'value': float(lsd_val),
#         'unit': 'RMSE Log-Spectral',
#         'note': 'Konsisten dengan evaluate_metrics_audio (Notebook)'
#     }
 
#     return metrics


# ==================== VISUALISASI ====================
def spectrogram_to_base64(spec, title="Spectrogram"):
    # Clip ke range [-80, 0] dB agar tampilan sama persis dengan Colab
    # (ref=1.0 bisa menghasilkan nilai > 0 dB, perlu di-clip dulu)
    spec_clipped = np.clip(spec, -80.0, 0.0)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        spec_clipped, sr=TARGET_SR, hop_length=HOP_LENGTH,
        x_axis='time', y_axis='mel', ax=ax,
        cmap='magma', fmin=FMIN, fmax=FMAX
    )
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title(title)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def audio_to_base64(audio_data, sr):
    buf = io.BytesIO()
    sf.write(buf, audio_data, sr, format='WAV')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def plot_spectrograms(original_spec, reconstructed_spec, output_path):
    # Clip ke [-80, 0] dB agar konsisten
    orig_clipped  = np.clip(original_spec,      -80.0, 0.0)
    recon_clipped = np.clip(reconstructed_spec, -80.0, 0.0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # --- GAMBAR 1: ASLI ---
    img1 = librosa.display.specshow(
        orig_clipped, sr=TARGET_SR, hop_length=HOP_LENGTH,
        x_axis='time', y_axis='mel', fmin=FMIN, fmax=FMAX,
        ax=axes[0],
        cmap='magma'
    )
    axes[0].set_title('Original Log-Mel Spectrogram')
    plt.colorbar(img1, ax=axes[0], format='%+2.0f dB')

    # --- GAMBAR 2: REKONSTRUKSI ---
    img2 = librosa.display.specshow(
        recon_clipped, sr=TARGET_SR, hop_length=HOP_LENGTH,
        x_axis='time', y_axis='mel', fmin=FMIN, fmax=FMAX,
        ax=axes[1],
        cmap='magma'
    )
    axes[1].set_title('VAE Reconstruction')
    plt.colorbar(img2, ax=axes[1], format='%+2.0f dB')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path

# ==================== ENDPOINTS ====================
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'device': str(device),
        'model_loaded': model is not None,
        'config': model_config
    })


@app.route('/info', methods=['GET'])
def get_model_info():
    return jsonify({
        'model': {
            'type': 'VAE',
            'latent_dim': model_config['latent_dim'],
            'input_shape': [1, N_MELS, 173]
        },
        'audio_params': {
            'sample_rate': TARGET_SR,
            'duration': TARGET_DURATION,
            'samples': TARGET_SAMPLES
        },
        'spectrogram_params': {
            'n_fft': N_FFT,
            'hop_length': HOP_LENGTH,
            'n_mels': N_MELS,
            'fmax': TARGET_SR / 2
        },
        'normalization': {
            'global_min': GLOBAL_MIN,
            'global_max': GLOBAL_MAX
        }
    })


@app.route('/process', methods=['POST'])
def process_audio():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not file.filename.lower().endswith('.wav'):
            return jsonify({'error': 'File must be WAV format'}), 400

        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        print(f"Processing file: {filename}")

        # 1. Preprocessing
        y_processed, preprocessing_steps = preprocess_audio(temp_path)

        # 2. Ekstraksi fitur (log-mel dB)
        log_mel_spec, extraction_steps = extract_log_mel(y_processed)

        # 3. Normalisasi ke [0, 1]
        spec_norm = normalize_spectrogram(log_mel_spec)
        noise_threshold = float(request.form.get('noise_threshold', 0.0))

        # 4. Rekonstruksi VAE (output juga [0, 1])
        recon_spec_norm, mu, logvar = reconstruct_vae(spec_norm, noise_threshold=noise_threshold)

        # 5. Denormalisasi kembali ke dB
        recon_spec_db = denormalize_spectrogram(recon_spec_norm)

        # 6. Hitung metrik pada domain spektrogram
        metrics = calculate_metrics(spec_norm, recon_spec_norm, log_mel_spec, recon_spec_db, mu, logvar)

        # 7. Konversi rekonstruksi ke audio
        reconstructed_audio = mel_to_audio_with_feedback(
            recon_spec_db,
            sr=TARGET_SR,
            y_original=y_processed,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )
        # reconstructed_audio = mel_to_audio(
        #     recon_spec_db,
        #     sr=TARGET_SR,
        #     n_fft=N_FFT,
        #     hop_length=HOP_LENGTH,
        #     preemphasis_coef=PREEMPHASIS_COEF,
        #     n_iter=128,
        #     apply_dsp=False  # <--- UBAH INI MENJADI TRUE
        # )

        # 10. Generate output
        output_base = os.path.join(OUTPUT_FOLDER, os.path.splitext(filename)[0])
        plot_path = f"{output_base}_comparison.png"
        plot_spectrograms(log_mel_spec, recon_spec_db, plot_path)

        recon_audio_path = f"{output_base}_reconstructed.wav"
        sf.write(recon_audio_path, reconstructed_audio, TARGET_SR)

        # 11. Prepare response
        response = {
            'success': True,
            'filename': filename,
            'preprocessing': {
                'steps': preprocessing_steps,
                'summary': {
                    'duration': f"{TARGET_DURATION}s",
                    'sample_rate': TARGET_SR,
                    'samples': TARGET_SAMPLES
                }
            },
            'feature_extraction': {
                'steps': extraction_steps,
                'summary': {
                    'n_mels': N_MELS,
                    'time_frames': log_mel_spec.shape[1],
                    'spectrogram_shape': list(log_mel_spec.shape)
                }
            },
            'spectrograms': {
                'original': spectrogram_to_base64(log_mel_spec, 'Original Spectrogram'),
                'reconstructed': spectrogram_to_base64(recon_spec_db, 'VAE Reconstruction')
            },
            'audio': {
                'original': audio_to_base64(y_processed, TARGET_SR),
                'reconstructed': audio_to_base64(reconstructed_audio, TARGET_SR)
            },
            'metrics': metrics,
            'latent_representation': {
                'mu': mu.cpu().numpy().tolist()[0],
                'logvar': logvar.cpu().numpy().tolist()[0]
            }
        }

        os.remove(temp_path)
        return jsonify(response)

    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/process_batch', methods=['POST'])
def process_batch():
    try:
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        files = request.files.getlist('files[]')
        if len(files) == 0:
            return jsonify({'error': 'No files selected'}), 400
        
        noise_threshold = float(request.form.get('noise_threshold', 0.20))

        results = []
        for file in files:
            if not file.filename.lower().endswith('.wav'):
                results.append({'filename': file.filename, 'success': False, 'error': 'Not a WAV file'})
                continue

            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(temp_path)

            try:
                y_processed, _ = preprocess_audio(temp_path)
                log_mel_spec, _ = extract_log_mel(y_processed)
                spec_norm = normalize_spectrogram(log_mel_spec)
                recon_spec_norm, mu, logvar = reconstruct_vae(spec_norm, noise_threshold=noise_threshold)
                recon_spec_db = denormalize_spectrogram(recon_spec_norm)
                metrics = calculate_metrics(spec_norm, recon_spec_norm, log_mel_spec, recon_spec_db, mu, logvar)
                reconstructed_audio = mel_to_audio_with_feedback(
                    recon_spec_db,
                    sr=TARGET_SR,
                    y_original=y_processed,
                    n_fft=N_FFT,
                    hop_length=HOP_LENGTH,
                )
                results.append({
                    'filename': file.filename,
                    'success': True,
                    'metrics': metrics,
                    'audio_reconstructed': audio_to_base64(reconstructed_audio, TARGET_SR)
                })
            except Exception as e:
                results.append({'filename': file.filename, 'success': False, 'error': str(e)})
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        return jsonify({
            'success': True,
            'total_files': len(files),
            'successful': sum(1 for r in results if r['success']),
            'failed': sum(1 for r in results if not r['success']),
            'results': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404


@app.teardown_appcontext
def cleanup_temp_files(exception=None):
    try:
        for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
            for fname in os.listdir(folder):
                fpath = os.path.join(folder, fname)
                if os.path.isfile(fpath):
                    os.remove(fpath)
    except Exception:
        pass


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)