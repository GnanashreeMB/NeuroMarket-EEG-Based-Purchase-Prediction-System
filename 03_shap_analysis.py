"""
NeuroMarket - SHAP Explainability Analysis
Shows mathematically which brain signals drive predictions
"""

import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import joblib
from data_loader import SEEDLoader
from feature_extractor import FeatureExtractor

print("="*70)
print("🧠 NEUROMARKET - SHAP EXPLAINABILITY ANALYSIS")
print("="*70)

# 1. Load data
print("\n[1/5] Loading data...")
loader = SEEDLoader()
trials = loader.load_trials()

# 2. Extract features
print("\n[2/5] Extracting features...")
extractor = FeatureExtractor()
X, y = extractor.extract_batch(trials)
feature_names = extractor.feature_names

print(f"Feature matrix: {X.shape}")
print(f"Number of features: {len(feature_names)}")

# 3. Train a model on ALL data (for SHAP analysis)
print("\n[3/5] Training final model for SHAP analysis...")
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    objective='binary:logistic',
    random_state=42
)
model.fit(X, y)

# 4. SHAP Analysis
print("\n[4/5] Running SHAP analysis (this may take 2-3 minutes)...")

# Create a sample of background data (for faster computation)
background = shap.sample(X, 100)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values for a subset
X_sample = X[:500]  # Analyze 500 samples
shap_values = explainer.shap_values(X_sample)

# 5. SHAP Summary Plot
print("\n[5/5] Generating SHAP plots...")

# Summary plot (most important features)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
plt.title('SHAP Feature Importance - Top 20 Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('models/shap_summary.png', dpi=150, bbox_inches='tight')
plt.show()
print("💾 SHAP summary saved to: models/shap_summary.png")

# 6. Bar plot of feature importance
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                  plot_type="bar", show=False)
plt.title('SHAP Feature Importance (Bar Chart)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('models/shap_bar.png', dpi=150, bbox_inches='tight')
plt.show()
print("💾 SHAP bar chart saved to: models/shap_bar.png")

# 7. Detailed analysis of top features
print("\n" + "="*70)
print("📊 TOP 10 FEATURES - DETAILED ANALYSIS")
print("="*70)

# Get mean absolute SHAP values
mean_shap = np.abs(shap_values).mean(axis=0)
top_indices = np.argsort(mean_shap)[-10:][::-1]

print("\n🔍 Top 10 Most Important Features:")
print("-" * 60)
print(f"{'Rank':<5} {'Feature':<25} {'Mean |SHAP|':<15} {'Band':<10} {'Interpretation'}")
print("-" * 60)

band_map = {
    'Delta': '😴 Deep relaxation',
    'Theta': '💭 Memory/Nostalgia',
    'Alpha': '😌 Relaxation',
    'Beta': '👀 Engagement',
    'Gamma': '🧠 Emotional'
}

for rank, idx in enumerate(top_indices, 1):
    feature = feature_names[idx]
    shap_value = mean_shap[idx]
    
    # Extract band from feature name
    if 'Delta' in feature: band = 'Delta'
    elif 'Theta' in feature: band = 'Theta'
    elif 'Alpha' in feature: band = 'Alpha'
    elif 'Beta' in feature: band = 'Beta'
    elif 'Gamma' in feature: band = 'Gamma'
    else: band = 'Other'
    
    interpretation = band_map.get(band, 'Neural activity')
    
    print(f"{rank:<5} {feature:<25} {shap_value:<15.4f} {band:<10} {interpretation}")

# 8. Gamma vs Beta comparison
print("\n" + "="*70)
print("⚡ GAMMA vs BETA - COMPARATIVE ANALYSIS")
print("="*70)

gamma_features = [i for i, name in enumerate(feature_names) if 'Gamma' in name]
beta_features = [i for i, name in enumerate(feature_names) if 'Beta' in name]
theta_features = [i for i, name in enumerate(feature_names) if 'Theta' in name]
alpha_features = [i for i, name in enumerate(feature_names) if 'Alpha' in name]

gamma_importance = np.mean([mean_shap[i] for i in gamma_features])
beta_importance = np.mean([mean_shap[i] for i in beta_features])
theta_importance = np.mean([mean_shap[i] for i in theta_features])
alpha_importance = np.mean([mean_shap[i] for i in alpha_features])

print(f"\n📊 Average Importance by Band:")
print(f"   🧠 GAMMA:   {gamma_importance:.4f} (Emotional connection)")
print(f"   👀 BETA:    {beta_importance:.4f} (Engagement)")
print(f"   💭 THETA:   {theta_importance:.4f} (Memory)")
print(f"   😌 ALPHA:   {alpha_importance:.4f} (Relaxation)")

# 9. Waterfall plot for a single prediction (BUY example)
print("\n" + "="*70)
print("🔍 SINGLE PREDICTION EXPLANATION")
print("="*70)

# Pick a BUY prediction example
buy_indices = np.where(y[:500] == 1)[0]
if len(buy_indices) > 0:
    idx = buy_indices[0]
    
    plt.figure(figsize=(12, 6))
    shap.waterfall_plot(shap.Explanation(values=shap_values[idx], 
                                         base_values=explainer.expected_value,
                                         data=X_sample[idx],
                                         feature_names=feature_names),
                       show=False)
    plt.title('SHAP Waterfall Plot - BUY Prediction Example', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('models/shap_waterfall_buy.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("💾 Waterfall plot saved to: models/shap_waterfall_buy.png")

# 10. Pick a NOT BUY prediction example
not_buy_indices = np.where(y[:500] == 0)[0]
if len(not_buy_indices) > 0:
    idx = not_buy_indices[0]
    
    plt.figure(figsize=(12, 6))
    shap.waterfall_plot(shap.Explanation(values=shap_values[idx], 
                                         base_values=explainer.expected_value,
                                         data=X_sample[idx],
                                         feature_names=feature_names),
                       show=False)
    plt.title('SHAP Waterfall Plot - NOT BUY Prediction Example', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('models/shap_waterfall_notbuy.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("💾 Waterfall plot saved to: models/shap_waterfall_notbuy.png")

# 11. Save SHAP values for later use
joblib.dump({
    'shap_values': shap_values,
    'feature_names': feature_names,
    'expected_value': explainer.expected_value,
    'X_sample': X_sample,
    'y_sample': y[:500]
}, 'models/shap_results.pkl')
print("\n💾 SHAP results saved to: models/shap_results.pkl")

print("\n" + "="*70)
print("✅ SHAP ANALYSIS COMPLETE!")
print("="*70)
print("\n📊 Generated files:")
print("   • models/shap_summary.png - Top features overview")
print("   • models/shap_bar.png - Feature importance bar chart")
print("   • models/shap_waterfall_buy.png - Single BUY explanation")
print("   • models/shap_waterfall_notbuy.png - Single NOT BUY explanation")
print("   • models/shap_results.pkl - Raw SHAP values")