"""
NeuroMarket - Model Training Module
Trains XGBoost to predict BUY (1) vs NOT BUY (0)
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class NeuroMarketTrainer:
    def __init__(self, model_dir='./models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def prepare_data(self, X, y, test_size=0.2):
        """Split data into train/test"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\n📊 Data split:")
        print(f"   Training: {len(self.X_train)} trials")
        print(f"   Testing: {len(self.X_test)} trials")
        print(f"   BUY in train: {sum(self.y_train)}")
        print(f"   NOT BUY in train: {len(self.y_train) - sum(self.y_train)}")
        
    def train_xgboost(self):
        """Train XGBoost classifier"""
        print("\n🚀 Training XGBoost model...")
        
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42
        )
        
        # Train the model
        self.model.fit(
            self.X_train, 
            self.y_train
        )
        
        # Predictions
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)
        
        # Accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"\n✅ Model trained!")
        print(f"   Test accuracy: {accuracy*100:.2f}%")
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(self.y_test, y_pred, 
                                   target_names=['NOT BUY', 'BUY']))
        
        return accuracy
    
    def show_feature_importance(self, top_n=20):
        """Show top features that predict purchase"""
        importance = self.model.feature_importances_
        
        # Get top features
        indices = np.argsort(importance)[::-1][:top_n]
        
        print(f"\n🔍 Top {top_n} Features for Purchase Prediction:")
        print("-" * 50)
        for i, idx in enumerate(indices):
            print(f"{i+1}. {self.feature_names[idx]}: {importance[idx]:.4f}")
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importance[indices][::-1])
        plt.yticks(range(top_n), [self.feature_names[i] for i in indices][::-1])
        plt.xlabel('Importance')
        plt.title('Top Features for Purchase Prediction')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.model_dir / 'feature_importance.png'
        plt.savefig(plot_path)
        print(f"\n📊 Plot saved to {plot_path}")
        
        return indices, importance
    
    def show_confusion_matrix(self):
        """Display confusion matrix"""
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['NOT BUY', 'BUY'],
                   yticklabels=['NOT BUY', 'BUY'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        # Save plot
        plot_path = self.model_dir / 'confusion_matrix.png'
        plt.savefig(plot_path)
        print(f"📊 Confusion matrix saved to {plot_path}")
        
        return cm
    
    def save_model(self):
        """Save trained model and metadata"""
        model_path = self.model_dir / 'xgboost_model.pkl'
        joblib.dump(self.model, model_path)
        
        # Save feature names
        np.save(self.model_dir / 'feature_names.npy', self.feature_names)
        
        print(f"\n💾 Model saved to {model_path}")
        
    def predict_single(self, features):
        """Predict for a single trial"""
        if self.model is None:
            print("Model not trained yet!")
            return None
        
        proba = self.model.predict_proba(features.reshape(1, -1))[0]
        pred = self.model.predict(features.reshape(1, -1))[0]
        
        return {
            'prediction': 'BUY' if pred == 1 else 'NOT BUY',
            'confidence': float(proba[1] if pred == 1 else proba[0]),
            'prob_buy': float(proba[1]),
            'prob_not_buy': float(proba[0])
        }


# Main execution
if __name__ == "__main__":
    from data_loader import SEEDLoader
    from feature_extractor import FeatureExtractor
    
    print("="*60)
    print("🧠 NEUROMARKET - Model Training")
    print("="*60)
    
    # 1. Load data
    print("\n[1/4] Loading data...")
    loader = SEEDLoader()
    trials = loader.load_trials()
    
    # 2. Extract features
    print("\n[2/4] Extracting features...")
    extractor = FeatureExtractor()
    X, y = extractor.extract_batch(trials)
    print(f"   Feature matrix: {X.shape}")
    
    # 3. Train model
    print("\n[3/4] Training model...")
    trainer = NeuroMarketTrainer()
    trainer.feature_names = extractor.feature_names
    trainer.prepare_data(X, y)
    accuracy = trainer.train_xgboost()
    
    # 4. Show results
    print("\n[4/4] Generating results...")
    trainer.show_feature_importance(top_n=15)
    trainer.show_confusion_matrix()
    trainer.save_model()
    
    print("\n" + "="*60)
    print(f"✅ TRAINING COMPLETE! Accuracy: {accuracy*100:.2f}%")
    print("="*60)