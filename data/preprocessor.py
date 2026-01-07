import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from .feature_extractor import extract_combined_fft_features

def extract_segments_around_spikes_hpo(signal, filename, variable_name, segment_size, threshold, 
                                     sample_rate, k_top, n_bands, n_bins, n_mfcc, crop_size):
    """Original segment extraction"""
    data = signal.get(variable_name)
    if data is None:
        print(f"Variable not found: {variable_name}")
        return pd.DataFrame()

    data = data.flatten()
    cleaned_data = data[16:]  # Original 16-sample removal
    cleaned_data = cleaned_data / np.max(np.abs(cleaned_data))  # Original normalization

    spike_indices = np.where(np.abs(cleaned_data) > threshold)[0]
    half = segment_size // 2
    spike_indices = spike_indices[(spike_indices > half) & (spike_indices < len(cleaned_data) - half)]

    # Original deduplication logic
    dedup_spikes = []
    last_idx = -segment_size
    for idx in spike_indices:
        if idx - last_idx >= segment_size:
            dedup_spikes.append(idx)
            last_idx = idx

    rows = []
    base_name = filename.split('.')[0]
    class_label = base_name.split('_')[0]
    
    for i, center in enumerate(dedup_spikes):
        start = int(center - crop_size//2)
        end = int(center + crop_size//2)
        if start < 0 or end > len(cleaned_data):
            continue
            
        segment = cleaned_data[start:end]
        features, feature_names = extract_combined_fft_features(
            segment, pad_to=crop_size, sample_rate=sample_rate,
            k_top=k_top, n_bands=n_bands, n_bins=n_bins, n_mfcc=n_mfcc
        )
        rows.append([base_name, class_label, i] + features.tolist())
    
    return pd.DataFrame(rows, columns=['file','class','segment'] + feature_names) if rows else pd.DataFrame()