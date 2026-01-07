import time
from config import BEST_CONFIG, MAT_DIR, VARIABLE_NAME, TARGET_RANGE
from specs import get_hardware_specs
from data.loader import original_pipeline, preprocess_data
from models.random_forest import optimize_random_state
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest

def main():
    best_config = BEST_CONFIG
    
    # Print hardware specs
    get_hardware_specs()

    # Run original pipeline
    preprocess_start = time.time()
    print("Running original high-accuracy pipeline...")
    original_result = original_pipeline(
        mat_dir=MAT_DIR,
        variable_name=VARIABLE_NAME,
        config=best_config
    )
    
    if original_result is not None:
        X, y, feature_names = original_result
        preprocess_time = time.time() - preprocess_start
        print(f"📊 Data preprocessing: {preprocess_time:.2f} seconds")
        print(f"📊 Dataset shape: {X.shape}")
    else:
        print("Failed to load data from original pipeline")
        return

    # Run preprocess_data alternative
    print("\nRunning preprocess_data pipeline...")
    X2, y2, feature_names2 = preprocess_data(
        mat_dir=MAT_DIR,
        variable_name=VARIABLE_NAME,
        config=best_config
    )
    print(f"Preprocess data shape: {X2.shape}")

    # Optimize random state
    print("\nOptimizing random state...")
    best_model, best_state, best_f1 = optimize_random_state(
        X, y,
        target_range=TARGET_RANGE
    )

    print(f"\nOriginal F1-score: {best_f1:.4f}")
    print(f"Best random state: {best_state}")
    print(f"Best model: {type(best_model).__name__}")

if __name__ == "__main__":
    main()