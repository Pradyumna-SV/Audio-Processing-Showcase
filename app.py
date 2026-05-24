import io
import hashlib
import importlib.util
from math import gcd
from pathlib import Path
import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import noisereduce as nr
import pandas as pd
import soundfile as sf
import streamlit as st
from scipy import linalg, signal


ALGORITHMS = [
    {
        "id": "highpass",
        "emoji": "↗",
        "name": "High-Pass Filter",
        "tag": "DSP",
        "category": "DSP",
        "subcategory": "Established",
        "tag_class": "tag-dsp",
        "desc": "Removes rumble and handling noise below 80 Hz; a simple conventional baseline.",
        "color": "#2563EB",
    },
    {
        "id": "bandpass",
        "emoji": "◫",
        "name": "Band-Pass Filter",
        "tag": "DSP",
        "category": "DSP",
        "subcategory": "Established",
        "tag_class": "tag-dsp",
        "desc": "Keeps the telephone-style speech band from 300 to 3,400 Hz.",
        "color": "#4F46E5",
    },
    {
        "id": "spectral_sub",
        "emoji": "−",
        "name": "Spectral Subtraction",
        "tag": "DSP",
        "category": "DSP",
        "subcategory": "Established",
        "tag_class": "tag-dsp",
        "desc": "Estimates a quiet noise profile and subtracts it in the spectrum, exposing classic musical-noise artifacts.",
        "color": "#7C3AED",
    },
    {
        "id": "wiener",
        "emoji": "≈",
        "name": "Wiener Filter",
        "tag": "DSP",
        "category": "DSP",
        "subcategory": "Established",
        "tag_class": "tag-dsp",
        "desc": "Smooths samples with a local minimum-error estimator, giving a familiar statistical DSP baseline.",
        "color": "#9333EA",
    },
    {
        "id": "kalman",
        "emoji": "◎",
        "name": "Kalman Speech Tracker",
        "tag": "DSP",
        "category": "DSP",
        "subcategory": "Established",
        "tag_class": "tag-dsp",
        "desc": "Tracks clean speech as a hidden changing state behind noisy microphone observations.",
        "color": "#10B981",
    },
    {
        "id": "emd_energy",
        "emoji": "∿",
        "name": "EMD IMF Energy Thresholding",
        "tag": "DSP",
        "category": "DSP",
        "subcategory": "Advanced",
        "tag_class": "tag-dsp",
        "desc": "Splits the recording into data-driven oscillatory modes and attenuates low-energy IMF regions; no fixed frequency bands are assumed.",
        "color": "#3B82F6",
    },
    {
        "id": "masking",
        "emoji": "◉",
        "name": "Auditory Masking Suppression",
        "tag": "DSP",
        "category": "DSP",
        "subcategory": "Advanced",
        "tag_class": "tag-dsp",
        "desc": "Uses a Bark-scale hearing model to reduce sounds hidden beneath stronger neighbors, favoring perceived clarity over raw energy reduction.",
        "color": "#8B5CF6",
    },
    {
        "id": "lpc_residual",
        "emoji": "△",
        "name": "LPC Residual Enhancement",
        "tag": "DSP",
        "category": "DSP",
        "subcategory": "Advanced",
        "tag_class": "tag-dsp",
        "desc": "Cleans the excitation source separately and synthesizes it through the estimated vocal-tract filter.",
        "color": "#EF4444",
    },
    {
        "id": "stockwell",
        "emoji": "⌁",
        "name": "Stockwell Transform Thresholding",
        "tag": "DSP",
        "category": "DSP",
        "subcategory": "Advanced",
        "tag_class": "tag-dsp",
        "desc": "Applies shrinkage in an S-transform representation with frequency-dependent windows and absolute phase.",
        "color": "#06B6D4",
    },
    {
        "id": "spectral_gate",
        "emoji": "▥",
        "name": "Adaptive Spectral Gate",
        "tag": "ML",
        "category": "ML",
        "subcategory": "Established",
        "tag_class": "tag-ml",
        "desc": "Estimates a changing noise threshold from the recording and gates frequency bins adaptively.",
        "color": "#059669",
    },
    {
        "id": "stationary_gate",
        "emoji": "▦",
        "name": "Stationary Spectral Gate",
        "tag": "ML",
        "category": "ML",
        "subcategory": "Established",
        "tag_class": "tag-ml",
        "desc": "Builds one stable spectral noise threshold, useful for fans and constant room hum.",
        "color": "#0D9488",
    },
    {
        "id": "pca_subspace",
        "emoji": "▱",
        "name": "PCA Spectral Subspace",
        "tag": "ML",
        "category": "ML",
        "subcategory": "Established",
        "tag_class": "tag-ml",
        "desc": "Compresses the magnitude spectrogram to dominant principal components and rejects low-energy variation.",
        "color": "#14B8A6",
    },
    {
        "id": "is_nmf",
        "emoji": "◇",
        "name": "Itakura-Saito NMF",
        "tag": "ML",
        "category": "ML",
        "subcategory": "Advanced",
        "tag_class": "tag-ml",
        "desc": "Learns voice-like spectral building blocks from this recording with IS-divergence factorization.",
        "color": "#F59E0B",
    },
    {
        "id": "ssa",
        "emoji": "▧",
        "name": "Singular Spectrum Reconstruction",
        "tag": "ML",
        "category": "ML",
        "subcategory": "Advanced",
        "tag_class": "tag-ml",
        "desc": "Rebuilds dominant low-rank structures in Hankel waveform matrices and discards unstructured tails.",
        "color": "#EC4899",
    },
    {
        "id": "deepfilternet",
        "emoji": "◆",
        "name": "DeepFilterNet",
        "tag": "DL",
        "category": "DL",
        "subcategory": "Established",
        "tag_class": "tag-dl",
        "desc": "Runs the published full-band neural deep-filtering model using bundled local weights.",
        "color": "#E11D48",
        "requires": "deepfilternet",
    },
    {
        "id": "df_masking",
        "emoji": "◈",
        "name": "DeepFilterNet + Auditory Masking",
        "tag": "DL",
        "category": "DL",
        "subcategory": "Advanced",
        "tag_class": "tag-dl",
        "desc": "DeepFilterNet output passed through auditory masking suppression.",
        "color": "#BE123C",
        "requires": "deepfilternet",
    },
]

CATEGORIES = ["DSP", "ML", "DL"]
SUBCATEGORIES = ["Established", "Advanced"]

PACKAGES = {
    "Baseline DSP": {
        "desc": "High-pass, band-pass, spectral subtraction, Wiener",
        "ids": ["highpass", "bandpass", "spectral_sub", "wiener"],
    },
    "Advanced DSP": {
        "desc": "EMD, masking, Stockwell, and related methods",
        "ids": ["emd_energy", "masking", "is_nmf", "ssa", "stockwell"],
    },
    "ML methods": {
        "desc": "Spectral gates, PCA, NMF, SSA",
        "ids": ["spectral_gate", "stationary_gate", "pca_subspace", "is_nmf", "ssa"],
    },
    "Deep learning": {
        "desc": "DeepFilterNet and hybrid cascade",
        "ids": ["deepfilternet", "df_masking"],
    },
    "All methods": {
        "desc": "Full list of installed techniques",
        "ids": [algorithm["id"] for algorithm in ALGORITHMS],
    },
}

DEEPFILTER_MODEL_DIR = Path(__file__).parent / "models" / "DeepFilterNet3"
DEEPFILTER_AVAILABLE = bool(
    importlib.util.find_spec("df")
    and importlib.util.find_spec("torch")
    and (DEEPFILTER_MODEL_DIR / "config.ini").exists()
)


def _finish(audio: np.ndarray, target_length: int) -> np.ndarray:
    out = np.nan_to_num(np.asarray(audio).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if out.size < target_length:
        out = np.pad(out, (0, target_length - out.size))
    out = out[:target_length]
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _stft(audio: np.ndarray, sr: int, frame_len: int = 512):
    hop = frame_len // 4
    working = np.pad(audio, (0, max(0, frame_len - len(audio))))
    return signal.stft(
        working,
        fs=sr,
        window="hann",
        nperseg=frame_len,
        noverlap=frame_len - hop,
        boundary="zeros",
        padded=True,
    )


def _istft(spectrum: np.ndarray, sr: int, length: int, frame_len: int = 512) -> np.ndarray:
    hop = frame_len // 4
    _, out = signal.istft(
        spectrum,
        fs=sr,
        window="hann",
        nperseg=frame_len,
        noverlap=frame_len - hop,
        input_onesided=True,
        boundary=True,
    )
    return _finish(out, length)


def highpass_filter(audio: np.ndarray, sr: int) -> np.ndarray:
    if len(audio) < 64:
        return _finish(audio, len(audio))
    sos = signal.butter(6, 80, btype="highpass", fs=sr, output="sos")
    return _finish(signal.sosfiltfilt(sos, audio), len(audio))


def bandpass_filter(audio: np.ndarray, sr: int) -> np.ndarray:
    if len(audio) < 64:
        return _finish(audio, len(audio))
    upper = min(3400, sr / 2 - 1)
    sos = signal.butter(6, [300, upper], btype="bandpass", fs=sr, output="sos")
    return _finish(signal.sosfiltfilt(sos, audio), len(audio))


def spectral_subtraction(audio: np.ndarray, sr: int) -> np.ndarray:
    length = len(audio)
    _, _, spectrum = _stft(audio, sr)
    magnitude = np.abs(spectrum)
    noise_frames = max(2, min(magnitude.shape[1], int(0.20 * sr / 128)))
    noise = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
    enhanced = np.maximum(magnitude - 1.5 * noise, 0.05 * magnitude)
    return _istft(enhanced * np.exp(1j * np.angle(spectrum)), sr, length)


def wiener_filter(audio: np.ndarray, sr: int) -> np.ndarray:
    if len(audio) < 5 or float(np.var(audio)) < 1e-12:
        return _finish(audio, len(audio))
    window = min(1023, max(3, len(audio) // 8))
    window = window if window % 2 else window - 1
    return _finish(signal.wiener(audio, mysize=window), len(audio))


def emd_energy_threshold(audio: np.ndarray, sr: int) -> np.ndarray:
    from PyEMD import EMD

    original_len = len(audio)
    if original_len < 32:
        return _finish(audio, original_len)

    work_sr = min(sr, 8000)
    if work_sr != sr:
        work = signal.resample_poly(audio, work_sr, sr).astype(np.float64)
    else:
        work = audio.astype(np.float64)

    imfs = EMD().emd(work, max_imf=7)
    if imfs.size == 0:
        return _finish(audio, original_len)

    enhanced = np.zeros_like(work)
    for idx, imf in enumerate(imfs):
        envelope = np.abs(signal.hilbert(imf))
        baseline = np.percentile(envelope, 35)
        threshold = baseline * (1.6 if idx < 2 else 1.15)
        gain = np.clip((envelope - threshold) / (envelope + 1e-8), 0.12, 1.0)
        kernel_size = min(31, len(gain) if len(gain) % 2 else len(gain) - 1)
        if kernel_size >= 3:
            gain = signal.medfilt(gain, kernel_size=kernel_size)
        enhanced += imf * gain

    if work_sr != sr:
        enhanced = signal.resample_poly(enhanced, sr, work_sr)
    return _finish(enhanced, original_len)


def auditory_masking_suppression(audio: np.ndarray, sr: int) -> np.ndarray:
    length = len(audio)
    frequencies, _, spectrum = _stft(audio, sr)
    power = np.abs(spectrum) ** 2 + 1e-12
    bark = 13.0 * np.arctan(0.00076 * frequencies) + 3.5 * np.arctan((frequencies / 7500.0) ** 2)
    band_centers = np.arange(0.0, 25.0, 1.0)
    membership = np.maximum(1.0 - np.abs(bark[:, None] - band_centers[None, :]), 0.0)
    membership /= np.maximum(membership.sum(axis=1, keepdims=True), 1e-12)
    band_power = membership.T @ power
    distances = band_centers[:, None] - band_centers[None, :]
    spreading = 10.0 ** (-(15.0 * np.maximum(distances, 0) + 27.0 * np.maximum(-distances, 0)) / 10.0)
    masked_bands = spreading @ band_power
    masking_power = membership @ masked_bands

    khz = np.maximum(frequencies / 1000.0, 0.02)
    absolute_db = 3.64 * khz ** -0.8 - 6.5 * np.exp(-0.6 * (khz - 3.3) ** 2) + 0.001 * khz**4
    reference = np.max(power, axis=0, keepdims=True)
    absolute_power = reference * 10.0 ** ((absolute_db[:, None] - 96.0) / 10.0)
    threshold = 0.07 * masking_power + absolute_power
    ratio = power / (threshold + 1e-12)
    gain = np.clip(np.sqrt(ratio), 0.1, 1.0)
    gain = signal.medfilt2d(gain, kernel_size=(3, 3))
    return _istft(spectrum * gain, sr, length)


def is_nmf_enhancement(audio: np.ndarray, sr: int) -> np.ndarray:
    length = len(audio)
    _, _, spectrum = _stft(audio, sr)
    observed = np.maximum(np.abs(spectrum) ** 2, 1e-10)
    rank = min(8, max(3, observed.shape[1] // 8))
    rng = np.random.default_rng(7)
    dictionary = rng.random((observed.shape[0], rank)) + 0.2
    activation = rng.random((rank, observed.shape[1])) + 0.2

    for _ in range(35):
        estimate = np.maximum(dictionary @ activation, 1e-10)
        activation *= (dictionary.T @ (observed / estimate**2)) / (dictionary.T @ (1.0 / estimate) + 1e-10)
        estimate = np.maximum(dictionary @ activation, 1e-10)
        dictionary *= ((observed / estimate**2) @ activation.T) / ((1.0 / estimate) @ activation.T + 1e-10)
        scale = np.maximum(dictionary.sum(axis=0), 1e-10)
        dictionary /= scale
        activation *= scale[:, None]

    flatness = np.exp(np.mean(np.log(dictionary + 1e-10), axis=0)) / np.mean(dictionary + 1e-10, axis=0)
    basis_energy = activation.sum(axis=1)
    ordered = np.lexsort((-basis_energy, flatness))
    speech_bases = ordered[: max(2, rank // 2)]
    speech_model = dictionary[:, speech_bases] @ activation[speech_bases, :]
    total_model = np.maximum(dictionary @ activation, 1e-10)
    gain = np.clip(speech_model / total_model, 0.08, 1.0)
    return _istft(spectrum * np.sqrt(gain), sr, length)


def adaptive_spectral_gate(audio: np.ndarray, sr: int) -> np.ndarray:
    if len(audio) < 512:
        return _finish(audio, len(audio))
    return _finish(nr.reduce_noise(y=audio, sr=sr, stationary=False, prop_decrease=0.85), len(audio))


def stationary_spectral_gate(audio: np.ndarray, sr: int) -> np.ndarray:
    if len(audio) < 512:
        return _finish(audio, len(audio))
    return _finish(nr.reduce_noise(y=audio, sr=sr, stationary=True, prop_decrease=0.85), len(audio))


def pca_spectral_subspace(audio: np.ndarray, sr: int) -> np.ndarray:
    length = len(audio)
    _, _, spectrum = _stft(audio, sr)
    magnitude = np.abs(spectrum)
    left, singular, right = linalg.svd(magnitude, full_matrices=False, check_finite=False)
    energy = np.cumsum(singular**2) / (np.sum(singular**2) + 1e-12)
    components = min(max(int(np.searchsorted(energy, 0.92)) + 1, 2), 12, len(singular))
    reconstructed = (left[:, :components] * singular[:components]) @ right[:components, :]
    gain = np.clip(reconstructed / (magnitude + 1e-10), 0.08, 1.0)
    return _istft(spectrum * gain, sr, length)


def kalman_speech_tracker(audio: np.ndarray, sr: int) -> np.ndarray:
    length = len(audio)
    if length == 0:
        return _finish(audio, length)
    if length < 3:
        return _finish(audio, length)
    rho = float(np.dot(audio[1:], audio[:-1]) / (np.dot(audio[:-1], audio[:-1]) + 1e-10))
    rho = float(np.clip(rho, -0.98, 0.98))
    innovations = audio[1:] - rho * audio[:-1]
    frame = max(32, sr // 50)
    frames = np.pad(audio, (0, (-length) % frame)).reshape(-1, frame)
    quiet_energy = np.percentile(np.mean(frames**2, axis=1), 20)
    measurement_noise = max(float(quiet_energy), 1e-7)
    process_noise = max(float(np.var(innovations)) - measurement_noise, measurement_noise * 0.03)

    estimate = 0.0
    variance = measurement_noise
    enhanced = np.empty(length, dtype=np.float64)
    for index, observed in enumerate(audio):
        predicted = rho * estimate
        predicted_variance = rho * rho * variance + process_noise
        gain = predicted_variance / (predicted_variance + measurement_noise)
        estimate = predicted + gain * (float(observed) - predicted)
        variance = (1.0 - gain) * predicted_variance
        enhanced[index] = estimate
    return _finish(enhanced, length)


def singular_spectrum_reconstruction(audio: np.ndarray, sr: int) -> np.ndarray:
    length = len(audio)
    block_size = min(2048, max(256, length))
    hop = block_size // 2
    embedding = min(80, block_size // 4)
    window = signal.windows.hann(block_size, sym=False)
    padded = np.pad(audio, (hop, block_size + hop))
    reconstructed = np.zeros_like(padded, dtype=np.float64)
    norm = np.zeros_like(padded, dtype=np.float64)

    for start in range(0, len(padded) - block_size + 1, hop):
        block = padded[start : start + block_size] * window
        trajectory = np.lib.stride_tricks.sliding_window_view(block, embedding).T
        left, singular, right = linalg.svd(trajectory, full_matrices=False, check_finite=False)
        energy = np.cumsum(singular**2) / (np.sum(singular**2) + 1e-12)
        components = min(max(int(np.searchsorted(energy, 0.88)) + 1, 2), 12)
        low_rank = (left[:, :components] * singular[:components]) @ right[:components, :]
        block_out = np.zeros(block_size)
        counts = np.zeros(block_size)
        for row in range(embedding):
            block_out[row : row + low_rank.shape[1]] += low_rank[row]
            counts[row : row + low_rank.shape[1]] += 1.0
        block_out /= np.maximum(counts, 1.0)
        reconstructed[start : start + block_size] += block_out * window
        norm[start : start + block_size] += window**2

    out = reconstructed[hop : hop + length] / np.maximum(norm[hop : hop + length], 1e-8)
    return _finish(out, length)


def _lpc_coefficients(frame: np.ndarray, order: int) -> np.ndarray:
    autocorrelation = signal.correlate(frame, frame, mode="full", method="fft")[len(frame) - 1 :]
    autocorrelation = autocorrelation[: order + 1]
    autocorrelation[0] += 1e-6
    try:
        prediction = linalg.solve_toeplitz(autocorrelation[:-1], autocorrelation[1:], check_finite=False)
    except linalg.LinAlgError:
        prediction = np.zeros(order)
    return np.concatenate(([1.0], -prediction))


def lpc_residual_enhancement(audio: np.ndarray, sr: int) -> np.ndarray:
    length = len(audio)
    frame_len = min(512, max(128, length))
    hop = frame_len // 2
    order = min(16, frame_len // 8)
    window = signal.windows.hann(frame_len, sym=False)
    padded = np.pad(audio, (hop, frame_len + hop))
    output = np.zeros_like(padded, dtype=np.float64)
    norm = np.zeros_like(padded, dtype=np.float64)

    for start in range(0, len(padded) - frame_len + 1, hop):
        frame = padded[start : start + frame_len] * window
        coefficients = _lpc_coefficients(frame, order)
        residual = signal.lfilter(coefficients, [1.0], frame)
        noise = np.median(np.abs(residual - np.median(residual))) / 0.6745 + 1e-9
        gain = np.clip(1.0 - (1.3 * noise / (np.abs(residual) + 1e-9)) ** 2, 0.12, 1.0)
        clean_residual = residual * gain
        synthesized = signal.lfilter([1.0], coefficients, clean_residual)
        output[start : start + frame_len] += synthesized * window
        norm[start : start + frame_len] += window**2

    out = output[hop : hop + length] / np.maximum(norm[hop : hop + length], 1e-8)
    return _finish(out, length)


def stockwell_thresholding(audio: np.ndarray, sr: int) -> np.ndarray:
    length = len(audio)
    block_size = min(512, max(128, length))
    hop = block_size // 2
    positive = np.arange(block_size // 2 + 1)
    offsets = np.fft.fftfreq(block_size) * block_size
    widths = np.maximum(positive, 1)[:, None]
    gaussian = np.exp(-2.0 * np.pi**2 * (offsets[None, :] / widths) ** 2)
    gaussian[0] = 1.0
    indices = (np.arange(block_size)[None, :] + positive[:, None]) % block_size
    window = signal.windows.hann(block_size, sym=False)
    padded = np.pad(audio, (hop, block_size + hop))
    output = np.zeros_like(padded, dtype=np.float64)
    norm = np.zeros_like(padded, dtype=np.float64)

    for start in range(0, len(padded) - block_size + 1, hop):
        block = padded[start : start + block_size] * window
        fourier = np.fft.fft(block)
        s_coefficients = np.fft.ifft(fourier[indices] * gaussian, axis=1)
        magnitudes = np.abs(s_coefficients)
        noise_floor = np.percentile(magnitudes[:, : max(8, block_size // 10)], 55, axis=1, keepdims=True)
        shrunk = s_coefficients * np.maximum(1.0 - 1.25 * noise_floor / (magnitudes + 1e-10), 0.08)
        recovered_positive = block_size * np.mean(shrunk, axis=1)
        recovered = np.fft.irfft(recovered_positive, n=block_size)
        output[start : start + block_size] += recovered * window
        norm[start : start + block_size] += window**2

    out = output[hop : hop + length] / np.maximum(norm[hop : hop + length], 1e-8)
    return _finish(out, length)


@st.cache_resource(show_spinner=False)
def _deepfilter_model():
    if not DEEPFILTER_AVAILABLE:
        raise RuntimeError("Install DeepFilterNet dependencies and provide bundled model weights to enable DL methods.")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*torchaudio\.backend\.common\.AudioMetaData.*")
        from df.enhance import init_df
        import df.logger as df_logger

    df_logger.get_commit_hash = lambda: None
    return init_df(model_base_dir=str(DEEPFILTER_MODEL_DIR), log_level="ERROR", log_file=None)


def deepfilter_enhancement(audio: np.ndarray, sr: int) -> np.ndarray:
    if not DEEPFILTER_AVAILABLE:
        raise RuntimeError("DeepFilterNet dependencies or bundled weights are unavailable.")
    import torch
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*torchaudio\.backend\.common\.AudioMetaData.*")
        from df.enhance import enhance

    model, state, _ = _deepfilter_model()
    target_sr = state.sr()
    if sr != target_sr:
        divisor = gcd(sr, target_sr)
        model_audio = signal.resample_poly(audio, target_sr // divisor, sr // divisor).astype(np.float32)
    else:
        model_audio = audio
    tensor = torch.from_numpy(model_audio).unsqueeze(0)
    output = enhance(model, state, tensor).squeeze().detach().cpu().numpy()
    if sr != target_sr:
        divisor = gcd(target_sr, sr)
        output = signal.resample_poly(output, sr // divisor, target_sr // divisor)
    return _finish(output, len(audio))


def deepfilter_masking_cascade(audio: np.ndarray, sr: int) -> np.ndarray:
    return auditory_masking_suppression(deepfilter_enhancement(audio, sr), sr)


PROC_MAP = {
    "highpass": highpass_filter,
    "bandpass": bandpass_filter,
    "spectral_sub": spectral_subtraction,
    "wiener": wiener_filter,
    "kalman": kalman_speech_tracker,
    "emd_energy": emd_energy_threshold,
    "masking": auditory_masking_suppression,
    "lpc_residual": lpc_residual_enhancement,
    "stockwell": stockwell_thresholding,
    "spectral_gate": adaptive_spectral_gate,
    "stationary_gate": stationary_spectral_gate,
    "pca_subspace": pca_spectral_subspace,
    "is_nmf": is_nmf_enhancement,
    "ssa": singular_spectrum_reconstruction,
    "deepfilternet": deepfilter_enhancement,
    "df_masking": deepfilter_masking_cascade,
}


def technique_available(algorithm: dict) -> bool:
    return algorithm.get("requires") != "deepfilternet" or DEEPFILTER_AVAILABLE


def to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def rms_db(audio: np.ndarray) -> float:
    return float(20.0 * np.log10(np.sqrt(np.mean(audio**2) + 1e-10)))


def make_comparison_figure(original: np.ndarray, results: list[tuple[str, np.ndarray, str]], sr: int):
    rows = 1 + len(results)
    figure = plt.figure(figsize=(14, 3.1 * rows), facecolor="#0F172A")
    grid = gridspec.GridSpec(rows, 2, figure=figure, wspace=0.04, hspace=0.55, left=0.07, right=0.97)
    times = np.arange(len(original)) / sr

    def plot_row(index: int, audio: np.ndarray, label: str, color: str):
        waveform = figure.add_subplot(grid[index, 0])
        waveform.plot(times, audio, color=color, linewidth=0.6)
        waveform.set_facecolor("#1E293B")
        waveform.set_title(label, color="white", fontsize=9, fontweight="bold")
        waveform.set_xlim(0, max(times[-1] if times.size else 0.01, 0.01))
        waveform.set_ylim(-1, 1)
        waveform.tick_params(colors="#94A3B8", labelsize=7)
        spectrogram = figure.add_subplot(grid[index, 1])
        segment = min(256, max(2, len(audio)))
        overlap = min(segment - 1, int(segment * 0.75))
        frequencies, t_spec, values = signal.spectrogram(audio, sr, nperseg=segment, noverlap=overlap)
        spectrogram.pcolormesh(t_spec, frequencies / 1000, 10 * np.log10(values + 1e-10), cmap="magma", vmin=-80, vmax=0)
        spectrogram.set_facecolor("#1E293B")
        spectrogram.set_ylim(0, min(8, sr / 2000))
        spectrogram.set_title("Spectrogram", color="#94A3B8", fontsize=8)
        spectrogram.tick_params(colors="#94A3B8", labelsize=7)
        spectrogram.set_ylabel("kHz", color="#94A3B8", fontsize=7)
        for axes in (waveform, spectrogram):
            for spine in axes.spines.values():
                spine.set_color("#334155")
        if index == rows - 1:
            waveform.set_xlabel("Time (s)", color="#94A3B8", fontsize=8)
            spectrogram.set_xlabel("Time (s)", color="#94A3B8", fontsize=8)

    plot_row(0, original, "Original", "#DC2626")
    for index, (algo_id, audio, label) in enumerate(results, start=1):
        color = next(item["color"] for item in ALGORITHMS if item["id"] == algo_id)
        plot_row(index, audio, label, color)
    return figure


def render_app() -> None:
    st.set_page_config(page_title="Voice Clarity Playground", page_icon="mic", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(
        """
        <style>
        .block-container { max-width: 1080px; padding-top: 3.5rem; padding-bottom: 4rem; }
        h1 { letter-spacing: -0.045em; font-weight: 650 !important; }
        [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }
        .hero { max-width: 600px; margin: 0 auto 2.25rem auto; text-align: center; }
        .hero p { color: #64748B; font-size: 1.03rem; }
        .surface { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 18px;
                   padding: 1rem 1.2rem; margin: .6rem 0 1.2rem; }
        .tag { display:inline-block; font-size:11px; font-weight:700; padding:2px 10px;
               border-radius:20px; letter-spacing:.5px; }
        .tag-dsp { background:#DBEAFE; color:#1D4ED8; }
        .tag-ml { background:#D1FAE5; color:#065F46; }
        .tag-dl { background:#FCE7F3; color:#9D174D; }
        .algo-desc { font-size:12px; color:#6B7280; margin-bottom:6px; }
        .meta { color:#64748B; font-size:12px; letter-spacing:.02em; text-transform:uppercase; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
          <h1>Voice Clarity Playground</h1>
          <p>Record a noisy voice sample, then choose how to clean it.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    recording = st.audio_input("Record", sample_rate=16000, label_visibility="collapsed")
    if recording is None:
        return

    original, sr = sf.read(io.BytesIO(recording.getvalue()), dtype="float32")
    if original.ndim > 1:
        original = original.mean(axis=1)
    original = _finish(original / max(float(np.max(np.abs(original))), 1e-8), len(original))
    audio_key = hashlib.sha256(recording.getvalue()).hexdigest()
    st.markdown("### Recording")
    st.audio(to_wav_bytes(original, sr), format="audio/wav")
    st.markdown("### Select techniques")
    package_name = st.selectbox(
        "Packaged solution",
        list(PACKAGES),
        index=0,
        format_func=lambda name: f"{name} - {PACKAGES[name]['desc']}",
    )
    requested_preset_ids = PACKAGES[package_name]["ids"]
    preset_ids = [identifier for identifier in requested_preset_ids if technique_available(next(a for a in ALGORITHMS if a["id"] == identifier))]
    preset_algorithms = [algorithm for algorithm in ALGORITHMS if algorithm["id"] in requested_preset_ids]

    selector_columns = st.columns(3)
    with selector_columns[0]:
        all_categories = st.checkbox("Select all categories", key=f"all_category_{package_name}")
        default_categories = list(dict.fromkeys(algorithm["category"] for algorithm in preset_algorithms))
        selected_categories = CATEGORIES if all_categories else st.multiselect(
            "Category",
            CATEGORIES,
            default=default_categories,
            key=f"category_{package_name}",
        )
    with selector_columns[1]:
        all_subcategories = st.checkbox("Select all subcategories", key=f"all_subcategory_{package_name}")
        default_subcategories = list(dict.fromkeys(algorithm["subcategory"] for algorithm in preset_algorithms))
        selected_subcategories = SUBCATEGORIES if all_subcategories else st.multiselect(
            "Subcategory",
            SUBCATEGORIES,
            default=default_subcategories,
            key=f"subcategory_{package_name}",
        )

    eligible = [
        algorithm
        for algorithm in ALGORITHMS
        if algorithm["category"] in selected_categories
        and algorithm["subcategory"] in selected_subcategories
        and technique_available(algorithm)
    ]
    unavailable = [
        algorithm
        for algorithm in ALGORITHMS
        if algorithm["category"] in selected_categories
        and algorithm["subcategory"] in selected_subcategories
        and not technique_available(algorithm)
    ]
    with selector_columns[2]:
        all_techniques = st.checkbox("Select all techniques", key=f"all_techniques_{package_name}")
        default_techniques = [algorithm["id"] for algorithm in eligible if algorithm["id"] in preset_ids]
        if all_techniques:
            selected_ids = [algorithm["id"] for algorithm in eligible]
            st.multiselect(
                "Technique",
                [algorithm["id"] for algorithm in eligible],
                default=selected_ids,
                format_func=lambda identifier: next(a["name"] for a in ALGORITHMS if a["id"] == identifier),
                disabled=True,
                key=f"all_display_{package_name}",
            )
        else:
            selected_ids = st.multiselect(
                "Technique",
                [algorithm["id"] for algorithm in eligible],
                default=default_techniques,
                format_func=lambda identifier: next(a["name"] for a in ALGORITHMS if a["id"] == identifier),
                key=f"techniques_{package_name}",
            )

    if unavailable:
        names = ", ".join(algorithm["name"] for algorithm in unavailable)
        st.caption(f"Deep learning methods unavailable ({names}). Requires PyTorch, deepfilternet, and models/DeepFilterNet3/.")

    with st.expander("Technique list"):
        for category in CATEGORIES:
            for subcategory in SUBCATEGORIES:
                techniques = [
                    algorithm for algorithm in ALGORITHMS
                    if algorithm["category"] == category and algorithm["subcategory"] == subcategory
                ]
                names = ", ".join(
                    f"{algorithm['name']}{'' if technique_available(algorithm) else ' (optional)'}"
                    for algorithm in techniques
                )
                st.markdown(f"**{category} / {subcategory}**  \n{names}")

    run = st.button("Run selected techniques", type="primary", disabled=not selected_ids, use_container_width=True)
    if run:
        processed_results = []
        for algorithm in ALGORITHMS:
            if algorithm["id"] not in selected_ids:
                continue
            with st.spinner(f"Running {algorithm['name']}..."):
                try:
                    result = PROC_MAP[algorithm["id"]](original, sr)
                    peak = float(np.max(np.abs(result)))
                    if peak > 0:
                        result = (result / peak * 0.95).astype(np.float32)
                    processed_results.append((algorithm["id"], result, algorithm["name"]))
                except Exception as error:
                    st.error(f"{algorithm['name']} failed: {error}")
        st.session_state["last_results"] = (audio_key, selected_ids, processed_results)
    stored_results = st.session_state.get("last_results")
    processed_results = (
        stored_results[2]
        if stored_results and stored_results[0] == audio_key and stored_results[1] == selected_ids
        else []
    )

    if processed_results:
        st.markdown("### Results")
        for algorithm_id, result, _ in processed_results:
            algorithm = next(item for item in ALGORITHMS if item["id"] == algorithm_id)
            st.markdown(
                f"**{algorithm['name']}** "
                f"<span class='tag {algorithm['tag_class']}'>{algorithm['category']} / {algorithm['subcategory']}</span>"
                f"<div class='algo-desc'>{algorithm['desc']}</div>",
                unsafe_allow_html=True,
            )
            st.audio(to_wav_bytes(result, sr), format="audio/wav")

        st.markdown("### Waveform & Spectrogram Comparison")
        figure = make_comparison_figure(original, processed_results, sr)
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)
        st.markdown("### Quick Energy Metrics")
        rows = [{"Algorithm": "Original", "Category": "Input", "Type": "-", "RMS Energy (dB)": f"{rms_db(original):.1f}"}]
        for algorithm_id, result, name in processed_results:
            algorithm = next(item for item in ALGORITHMS if item["id"] == algorithm_id)
            rows.append({
                "Algorithm": name,
                "Category": algorithm["category"],
                "Type": algorithm["subcategory"],
                "RMS Energy (dB)": f"{rms_db(result):.1f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    render_app()
