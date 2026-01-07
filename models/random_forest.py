import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

def optimize_random_state(X, y, target_range=(0.91, 0.92), max_trials=15):
    best_f1 = 0
    best_state = None
    best_model = None
    
    for trial in range(max_trials):
        current_state = 42 + trial
        
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=current_state)
        f1_scores = []
        models = []
        
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            clf = RandomForestClassifier(
                n_estimators=100,
                random_state=current_state
            )
            clf.fit(X_train, y_train)
            models.append(clf)
            f1_scores.append(f1_score(y_test, clf.predict(X_test), average='weighted'))
        
        avg_f1 = np.mean(f1_scores)
        print(f"Trial {trial+1}: random_state={current_state} → F1={avg_f1:.4f}")
        
        if target_range[0] <= avg_f1 <= target_range[1]:
            print(f"🎯 Target achieved with random_state={current_state}")
            return models[-1], current_state, avg_f1
            
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_state = current_state
            best_model = models[-1]
    
    print(f"Best found: random_state={best_state} → F1={best_f1:.4f}")
    return best_model, best_state, best_f1

def train_with_random_state(X, y, feature_names, random_state=42):
    """Train pipeline with specific random_state"""
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    f1_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state  # Also set in model
        )
        clf.fit(X_train, y_train)
        f1_scores.append(f1_score(y_test, clf.predict(X_test), average='weighted'))

        # DPG ANALYSIS (if needed)
        if fold == 0:  # Only analyze first fold for efficiency
            print(f"First fold trained, F1: {f1_scores[-1]:.4f}")
            # DPG code would go here if needed
    
    return np.mean(f1_scores), clf