"""
NeuroMarket - Leave-One-Subject-Out Cross Validation
Tests model performance on NEVER-BEFORE-SEEN individuals
"""

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
import xgboost as xgb
import joblib
from data_loader import SEEDLoader
from feature_extractor import FeatureExtractor

print("="*70)
print("🧠 NEUROMARKET - LEAVE-ONE-SUBJECT-OUT VALIDATION")
print("="*70)

# 1. Load data
print("\n[1/5] Loading data...")
loader = SEEDLoader()
trials = loader.load_trials()

# 2. Extract features
print("\n[2/5] Extracting features...")
extractor = FeatureExtractor()
X, y = extractor.extract_batch(trials)

# 3. Get subject IDs for grouping
print("\n[3/5] Preparing subject groups...")
groups = np.array([int(t['subject']) for t in trials])
unique_subjects = np.unique(groups)
print(f"Total trials: {len(trials)}")
print(f"Unique subjects: {len(unique_subjects)}")
print(f"Subject IDs: {unique_subjects}")

# 4. Leave-One-Subject-Out Cross-Validation
print("\n[4/5] Running Leave-One-Subject-Out Validation...")
print("-" * 70)

logo = LeaveOneGroupOut()
fold_results = []

for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
    # Get the test subject
    test_subject = groups[test_idx[0]]
    
    # Split data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"\nFold {fold+1}: Testing on Subject {test_subject}")
    print(f"   Training: {len(X_train)} trials")
    print(f"   Testing:  {len(X_test)} trials")
    
    # Train model
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    
    print(f"   Accuracy:  {acc*100:.2f}%")
    print(f"   ROC-AUC:   {auc:.3f}")
    print(f"   F1-Score:  {f1:.3f}")
    
    # Store results
    fold_results.append({
        'subject': test_subject,
        'accuracy': acc,
        'auc': auc,
        'f1': f1,
        'y_true': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba
    })

# 5. Summary Statistics
print("\n" + "="*70)
print("[5/5] FINAL RESULTS - Across ALL Subjects")
print("="*70)

accuracies = [r['accuracy'] for r in fold_results]
aucs = [r['auc'] for r in fold_results]
f1s = [r['f1'] for r in fold_results]

print(f"\n📊 Performance Across {len(unique_subjects)} Subjects:")
print(f"   Accuracy:  {np.mean(accuracies)*100:.2f}% (±{np.std(accuracies)*100:.2f})")
print(f"   ROC-AUC:   {np.mean(aucs):.3f} (±{np.std(aucs):.3f})")
print(f"   F1-Score:  {np.mean(f1s):.3f} (±{np.std(f1s):.3f})")

# Best and worst subjects
best_idx = np.argmax(accuracies)
worst_idx = np.argmin(accuracies)

print(f"\n📈 Best performing subject: Subject {fold_results[best_idx]['subject']}")
print(f"    Accuracy: {accuracies[best_idx]*100:.2f}%")
print(f"\n📉 Worst performing subject: Subject {fold_results[worst_idx]['subject']}")
print(f"    Accuracy: {accuracies[worst_idx]*100:.2f}%")

# Save results
results_summary = {
    'mean_accuracy': np.mean(accuracies),
    'std_accuracy': np.std(accuracies),
    'mean_auc': np.mean(aucs),
    'std_auc': np.std(aucs),
    'mean_f1': np.mean(f1s),
    'std_f1': np.std(f1s),
    'per_subject': fold_results
}

joblib.dump(results_summary, 'models/loso_results.pkl')
print(f"\n💾 Results saved to models/loso_results.pkl")

print("\n" + "="*70)
print("✅ VALIDATION COMPLETE!")
print("="*70)