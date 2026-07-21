"""
NeuroMarket - Feature Extractor Module
Since SEED data already has features, we just flatten them
"""

import numpy as np

class FeatureExtractor:
    def __init__(self):
        self.feature_names = []
        
    def extract_features(self, trial_eeg):
        """
        Convert (5, 62) EEG features to flat array of 310 features
        
        Args:
            trial_eeg: numpy array of shape (5, 62)
                     5 frequency bands × 62 channels
        
        Returns:
            features: numpy array of shape (310,)
            feature_names: list of feature names
        """
        # Flatten the 5×62 matrix to 310 features
        features = trial_eeg.flatten()
        
        # Generate feature names for reference
        if not self.feature_names:
            band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
            self.feature_names = []
            
            # Create names like: Delta_CH1, Delta_CH2, ..., Gamma_CH62
            for band_idx, band in enumerate(band_names):
                for ch_idx in range(1, 63):
                    self.feature_names.append(f"{band}_CH{ch_idx}")
        
        return features
    
    def extract_batch(self, trials_list):
        """
        Extract features for multiple trials
        
        Args:
            trials_list: list of trial dictionaries from data_loader
        
        Returns:
            X: numpy array of shape (n_trials, 310)
            y: numpy array of shape (n_trials,)
        """
        X = []
        y = []
        
        for trial in trials_list:
            features = self.extract_features(trial['eeg'])
            X.append(features)
            y.append(trial['purchase_intent'])
        
        return np.array(X), np.array(y)


# Quick test
if __name__ == "__main__":
    # Test with sample data
    from data_loader import SEEDLoader
    
    # Load data
    loader = SEEDLoader()
    trials = loader.load_trials()
    
    # Extract features
    extractor = FeatureExtractor()
    X, y = extractor.extract_batch(trials[:5])  # Test with first 5 trials
    
    print(f"\n✅ Feature extraction test:")
    print(f"   Input shape (per trial): (5, 62)")
    print(f"   Output shape: {X.shape}")
    print(f"   Features per trial: {X.shape[1]}")
    print(f"   Labels: {y}")
    print(f"\n   First 5 feature names: {extractor.feature_names[:5]}")