"""
NeuroMarket - Confusion Matrix and Detailed Metrics
Shows exactly how the model performs on BUY vs NOT BUY
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import joblib
import pandas as pd

print("="*70)
print("📊 NEUROMARKET - CONFUSION MATRIX & DETAILED METRICS")
print("="*70)

# Load LOSO results
results = joblib.load('models/loso_results.pkl')
per_subject = results['per_subject']

# Aggregate all predictions
all_y_true = []
all_y_pred = []
all_y_proba = []

for fold in per_subject:
    all_y_true.extend(fold['y_true'])
    all_y_pred.extend(fold['y_pred'])
    all_y_proba.extend(fold['y_proba'])

all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)
all_y_proba = np.array(all_y_proba)

print(f"\n📈 Total predictions analyzed: {len(all_y_true)}")

# 1. Classification Report
print("\n" + "="*70)
print("📋 CLASSIFICATION REPORT - ALL SUBJECTS COMBINED")
print("="*70)
print("\n")

# Generate report
report = classification_report(all_y_true, all_y_pred, 
                              target_names=['NOT BUY', 'BUY'],
                              output_dict=True)

# Print nicely formatted
print(f"{'':15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
print("-" * 60)
print(f"{'NOT BUY':15} {report['NOT BUY']['precision']:10.3f} {report['NOT BUY']['recall']:10.3f} "
      f"{report['NOT BUY']['f1-score']:10.3f} {report['NOT BUY']['support']:10.0f}")
print(f"{'BUY':15} {report['BUY']['precision']:10.3f} {report['BUY']['recall']:10.3f} "
      f"{report['BUY']['f1-score']:10.3f} {report['BUY']['support']:10.0f}")
print("-" * 60)
print(f"{'Accuracy':15} {'':>10} {'':>10} {report['accuracy']:10.3f} {len(all_y_true):10.0f}")

# 2. Confusion Matrix
print("\n" + "="*70)
print("🔢 CONFUSION MATRIX")
print("="*70)

cm = confusion_matrix(all_y_true, all_y_pred)

# Create a nice DataFrame for display
cm_df = pd.DataFrame(cm, 
                     index=['Actual NOT BUY', 'Actual BUY'],
                     columns=['Predicted NOT BUY', 'Predicted BUY'])
print("\n", cm_df)
print("\n")

# Calculate percentages
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
cm_percent_df = pd.DataFrame(cm_percent, 
                            index=['Actual NOT BUY', 'Actual BUY'],
                            columns=['Predicted NOT BUY %', 'Predicted BUY %'])
print("\nPercentages:")
print(cm_percent_df.round(1))

# 3. Plot Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=['Predicted NOT BUY', 'Predicted BUY'],
           yticklabels=['Actual NOT BUY', 'Actual BUY'])
plt.title('Confusion Matrix - All Subjects Combined', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=150)
plt.show()
print("\n💾 Confusion matrix saved to: models/confusion_matrix.png")

# 4. ROC Curve
print("\n" + "="*70)
print("📈 ROC CURVE")
print("="*70)

fpr, tpr, _ = roc_curve(all_y_true, all_y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('models/roc_curve.png', dpi=150)
plt.show()
print(f"💾 ROC curve saved to: models/roc_curve.png")
print(f"📊 AUC Score: {roc_auc:.3f}")

# 5. Per-Subject Performance Summary
print("\n" + "="*70)
print("👤 PER-SUBJECT PERFORMANCE SUMMARY")
print("="*70)

subject_summary = []
for fold in per_subject:
    subject_summary.append({
        'Subject': fold['subject'],
        'Accuracy': f"{fold['accuracy']*100:.1f}%",
        'AUC': f"{fold['auc']:.3f}",
        'F1': f"{fold['f1']:.3f}"
    })

summary_df = pd.DataFrame(subject_summary)
print("\n", summary_df.to_string(index=False))

# Save summary to CSV
summary_df.to_csv('models/subject_performance.csv', index=False)
print("\n💾 Per-subject performance saved to: models/subject_performance.csv")

# 6. Final Summary Statistics
print("\n" + "="*70)
print("🎯 FINAL SUMMARY STATISTICS")
print("="*70)

print(f"""
📊 Overall Performance:
   • Accuracy:  {report['accuracy']*100:.1f}%
   • ROC-AUC:   {roc_auc:.3f}
   • Precision (BUY): {report['BUY']['precision']:.3f}
   • Recall (BUY):    {report['BUY']['recall']:.3f}
   • F1-Score (BUY):  {report['BUY']['f1-score']:.3f}

🔍 Confusion Matrix Insights:
   • True Negatives (correct NOT BUY): {cm[0,0]}
   • False Positives (wrongly predicted BUY): {cm[0,1]}
   • False Negatives (missed BUY): {cm[1,0]}
   • True Positives (correct BUY): {cm[1,1]}

📉 Error Analysis:
   • False Positive Rate: {cm[0,1]/(cm[0,0]+cm[0,1])*100:.1f}%
   • False Negative Rate: {cm[1,0]/(cm[1,0]+cm[1,1])*100:.1f}%
""")

print("="*70)
print("✅ ANALYSIS COMPLETE!")
print("="*70)