import os
import glob
import pandas as pd
from scipy.io import loadmat
from .preprocessor import extract_segments_around_spikes_hpo
from sklearn.ensemble import RandomForestClassifier, IsolationForest

def original_pipeline(mat_dir, variable_name, config):
    """Exact original pipeline implementation"""
    segment_size, threshold, sample_rate, k_top, n_bands, n_bins, n_mfcc, crop_size = config
    
    all_dfs = []
    for filepath in glob.glob(os.path.join(mat_dir, "*.mat")):
        try:
            signal = loadmat(filepath)
            df = extract_segments_around_spikes_hpo(
                signal, os.path.basename(filepath), 
                variable_name, segment_size, threshold,
                sample_rate, k_top, n_bands, n_bins, n_mfcc, crop_size
            )
            if not df.empty:
                all_dfs.append(df)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    if not all_dfs:
        print("No valid data extracted")
        return None

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df = full_df[full_df['class'].isin(['LC','CV'])].reset_index(drop=True)
    
    features = full_df.iloc[:, 3:]
    labels = full_df['class']
    

    print(f"Total files processed: {len(all_dfs)}")
    print(f"Total nut segments: {len(full_df)}")
    print(f"Classes distribution:\n{full_df['class'].value_counts()}")
    
    # Original outlier removal
    X = features.dropna().to_numpy()
    y = labels[features.dropna().index].to_numpy()
    
    iso = IsolationForest(n_estimators=100, contamination=0.10, random_state=42)
    inliers = iso.fit_predict(X) == 1
    X = X[inliers]
    y = y[inliers]
    
    return X, y, features.columns.tolist()

def preprocess_data(mat_dir, variable_name, config):
    """Your complete data preprocessing pipeline"""
    segment_size, threshold, sample_rate, k_top, n_bands, n_bins, n_mfcc, crop_size = config
    
    all_dfs = []
    for filepath in glob.glob(os.path.join(mat_dir, "*.mat")):
        try:
            signal = loadmat(filepath)
            df = extract_segments_around_spikes_hpo(
                signal, os.path.basename(filepath), 
                variable_name, segment_size, threshold,
                sample_rate, k_top, n_bands, n_bins, n_mfcc, crop_size
            )
            if not df.empty:
                all_dfs.append(df)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    if not all_dfs:
        raise ValueError("No valid data extracted")

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df = full_df[full_df['class'].isin(['LC','CV'])].reset_index(drop=True)
    
    features = full_df.iloc[:, 3:].dropna()
    labels = full_df.loc[features.index, 'class']
    X = features.to_numpy()
    y = labels.to_numpy()
    feature_names = features.columns.tolist()

    # Outlier removal
    iso = IsolationForest(n_estimators=100, contamination=0.10, random_state=42)
    inliers = iso.fit_predict(X) == 1
    X = X[inliers]
    y = y[inliers]
    
    return X, y, feature_names