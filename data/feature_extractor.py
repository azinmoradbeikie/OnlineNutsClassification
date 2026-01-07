import numpy as np
import librosa
from scipy.signal import get_window

def apply_window(segment, window_type="hann"):
    window = get_window(window_type, len(segment))
    return segment * window

def extract_combined_fft_features(segment, pad_to, sample_rate, k_top, n_bands, n_bins, n_mfcc, window_type="hann"):
    """Original feature extraction with all parameters"""
    feature_names = []
    
    # Original preprocessing
    segment = apply_window(segment, window_type)
    if len(segment) < pad_to:
        segment = np.pad(segment, (0, pad_to - len(segment)), mode='constant')

    # Original FFT computation
    fft_vals = np.abs(np.fft.rfft(segment))
    fft_log = np.log1p(fft_vals)
    freqs = np.fft.rfftfreq(pad_to, d=1.0/sample_rate)
    power = fft_vals**2

    # Original feature extraction logic
    topk_idx = np.argsort(fft_log)[-k_top:][::-1]
    topk_mag = fft_log[topk_idx]
    
    band_energies = []
    band_size = len(power) // n_bands
    for i in range(n_bands):
        band_energies.append(np.sum(power[i*band_size:(i+1)*band_size]))
    
    total_energy = np.sum(power) + 1e-10
    centroid = np.sum(freqs * power) / total_energy
    bandwidth = np.sqrt(np.sum((freqs - centroid)**2 * power) / total_energy)
    flatness = np.exp(np.mean(np.log(fft_vals + 1e-10))) / (np.mean(fft_vals) + 1e-10)
    rolloff = freqs[np.where(np.cumsum(power) >= 0.85 * total_energy)[0][0]]
    
    fft_bins = []
    bin_size = len(fft_log) // n_bins
    for i in range(n_bins):
        fft_bins.append(np.mean(fft_log[i*bin_size:(i+1)*bin_size]))
    
    try:
        mfccs = librosa.feature.mfcc(y=segment.astype(np.float32), sr=sample_rate, n_mfcc=n_mfcc)
        mfcc_features = np.mean(mfccs, axis=1)
    except Exception as e:
        print(f"MFCC error: {e}")
        mfcc_features = np.zeros(n_mfcc)

    # Original feature concatenation order
    combined = np.concatenate([
        topk_idx, topk_mag, 
        band_energies, 
        [centroid, bandwidth, flatness, rolloff],
        fft_bins, 
        mfcc_features
    ])
    
    feature_names = (
        [f"topk_idx_{i}" for i in range(k_top)] +
        [f"topk_mag_{i}" for i in range(k_top)] +
        [f"band_energy_{i}" for i in range(n_bands)] +
        ["centroid", "bandwidth", "flatness", "rolloff"] +
        [f"fft_bin_{i}" for i in range(n_bins)] +
        [f"mfcc_{i}" for i in range(n_mfcc)]
    )
    
    return combined, feature_names