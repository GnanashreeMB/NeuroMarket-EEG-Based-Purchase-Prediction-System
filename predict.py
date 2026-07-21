"""
NeuroMarket - Prediction Module
Uses trained model to predict purchase intent
"""

import numpy as np
import joblib
from pathlib import Path

class NeuroMarketPredictor:
    def __init__(self, model_dir='./models'):
        self.model_dir = Path(model_dir)
        self.model = joblib.load(self.model_dir / 'xgboost_model.pkl')
        self.feature_names = np.load(self.model_dir / 'feature_names.npy', allow_pickle=True)
        
    def predict(self, features):
        """
        Predict purchase intent from features
        
        Args:
            features: numpy array of shape (310,) or (n, 310)
        
        Returns:
            dict with prediction and confidence
        """
        # Ensure 2D
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Predict
        pred = self.model.predict(features)[0]
        proba = self.model.predict_proba(features)[0]
        
        return {
            'prediction': 'BUY' if pred == 1 else 'NOT BUY',
            'confidence': float(proba[1] if pred == 1 else proba[0]),
            'prob_buy': float(proba[1]),
            'prob_not_buy': float(proba[0])
        }
    
    def explain_prediction(self, features, top_n=5):
        """
        Explain why the model made this prediction
        Shows top features that influenced the decision
        """
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Get prediction
        result = self.predict(features)
        
        # Get feature importance for this prediction
        # For tree models, we can use the feature importance scores
        importance = self.model.feature_importances_
        
        # Get top features
        indices = np.argsort(importance)[::-1][:top_n]
        
        explanation = {
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'top_features': []
        }
        
        for idx in indices:
            feature_name = self.feature_names[idx]
            importance_val = importance[idx]
            
            # Interpret the feature
            band = feature_name.split('_')[0]
            channel = feature_name.split('_')[1]
            
            interpretation = self._interpret_feature(band, importance_val)
            
            explanation['top_features'].append({
                'feature': feature_name,
                'importance': float(importance_val),
                'band': band,
                'channel': channel,
                'interpretation': interpretation
            })
        
        return explanation
    
    def _interpret_feature(self, band, importance):
        """Convert band to human-readable meaning"""
        interpretations = {
            'Gamma': '🧠 Emotional connection to product',
            'Beta': '👀 Active engagement and attention',
            'Alpha': '😌 Relaxed processing',
            'Theta': '💭 Memory recall (nostalgia)',
            'Delta': '😴 Deep unconscious processing'
        }
        return interpretations.get(band, 'Neural activity')


# Test
if __name__ == "__main__":
    from data_loader import SEEDLoader
    from feature_extractor import FeatureExtractor
    
    # Load a sample trial
    loader = SEEDLoader()
    trials = loader.load_trials()
    
    # Extract features for first trial
    extractor = FeatureExtractor()
    X, y = extractor.extract_batch(trials[:1])
    
    # Predict
    predictor = NeuroMarketPredictor()
    result = predictor.predict(X[0])
    
    print("\n🔮 PREDICTION RESULT:")
    print(f"   {result['prediction']} with {result['confidence']*100:.1f}% confidence")
    
    # Explain
    explanation = predictor.explain_prediction(X[0])
    print("\n📊 WHY THIS PREDICTION:")
    for feat in explanation['top_features']:
        print(f"   {feat['feature']}: {feat['interpretation']} (importance: {feat['importance']:.4f})")