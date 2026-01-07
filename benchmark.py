import time
from ..data.feature_extractor import extract_combined_fft_features

def benchmark_feature_extraction(segment, config):
    """Time feature extraction for one segment"""
    start_time = time.perf_counter()
    
    features, feature_names = extract_combined_fft_features(
        segment, 
        pad_to=config[7],  # crop_size
        sample_rate=config[2], 
        k_top=config[3], 
        n_bands=config[4], 
        n_bins=config[5], 
        n_mfcc=config[6]
    )
    
    extraction_time = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
    return features, feature_names, extraction_time

def benchmark_inference(model, X_sample):
    """Time inference for one sample"""
    start_time = time.perf_counter()
    prediction = model.predict(X_sample.reshape(1, -1))
    inference_time = (time.perf_counter() - start_time) * 1000  # milliseconds
    return prediction, inference_time

def benchmark_throughput(model, X_test, batch_size=100):
    """Measure throughput (samples per second)"""
    start_time = time.perf_counter()
    predictions = model.predict(X_test[:batch_size])
    total_time = time.perf_counter() - start_time
    
    throughput = batch_size / total_time  # samples per second
    return throughput