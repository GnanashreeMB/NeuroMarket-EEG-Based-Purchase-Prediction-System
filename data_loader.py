"""
NeuroMarket - Data Loader Module for SEED Dataset (NPZ format)
"""

import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class SEEDLoader:
    def __init__(self, data_path='./data'):
        self.data_path = Path(data_path)
        self.raw_path = self.data_path / 'raw'
        self.processed_path = self.data_path / 'processed'
        
        # Create directories
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
    def check_files_exist(self):
        """Check if all required NPZ files exist"""
        required_files = ['DatasetCaricatoNoImage.npz', 
                         'LabelsNoImage.npz', 
                         'SubjectsNoImage.npz']
        
        for file in required_files:
            if not (self.raw_path / file).exists():
                print(f"Missing: {file}")
                return False
        return True
    
    def load_trials(self):
        """Load all trials and map labels to purchase intent"""
        
        # Load the files
        dataset_path = self.raw_path / 'DatasetCaricatoNoImage.npz'
        labels_path = self.raw_path / 'LabelsNoImage.npz'
        subjects_path = self.raw_path / 'SubjectsNoImage.npz'
        
        print("Loading NPZ files...")
        eeg_data = np.load(dataset_path)['arr_0']
        labels = np.load(labels_path)['arr_0']
        subjects = np.load(subjects_path)['arr_0']
        
        print(f"\n📊 Data shapes:")
        print(f"   EEG data: {eeg_data.shape}")
        print(f"   Labels: {labels.shape}")
        print(f"   Subjects: {subjects.shape}")
        
        # Label mapping:
        # 2 = positive emotion → BUY (1)
        # 1 = negative emotion → NOT BUY (0)
        # 0 = neutral → SKIP
        
        all_trials = []
        buy_count = 0
        not_buy_count = 0
        skip_count = 0
        
        for trial_idx in range(len(labels)):
            label_val = labels[trial_idx]
            
            # Map to purchase intent
            if label_val == 2:  # Positive → BUY
                purchase_intent = 1
                buy_count += 1
            elif label_val == 1:  # Negative → NOT BUY
                purchase_intent = 0
                not_buy_count += 1
            else:  # label_val == 0 → Neutral, skip
                skip_count += 1
                continue
            
            trial = {
                'trial_idx': trial_idx,
                'eeg': eeg_data[trial_idx],  # Shape: (5, 62)
                'sampling_rate': 200,
                'label_value': int(label_val),
                'subject': str(subjects[trial_idx]),
                'purchase_intent': purchase_intent
            }
            all_trials.append(trial)
        
        print(f"\n✅ Trials loaded:")
        print(f"   BUY trials (label=2): {buy_count}")
        print(f"   NOT BUY trials (label=1): {not_buy_count}")
        print(f"   Neutral trials skipped (label=0): {skip_count}")
        print(f"   Total usable trials: {len(all_trials)}")
        
        if len(all_trials) > 0:
            print(f"\nSample trial:")
            print(f"   EEG shape: {all_trials[0]['eeg'].shape}")
            print(f"   Label value: {all_trials[0]['label_value']}")
            print(f"   Purchase: {'BUY' if all_trials[0]['purchase_intent']==1 else 'NOT BUY'}")
        
        return all_trials


# Quick test
if __name__ == "__main__":
    loader = SEEDLoader()
    
    if loader.check_files_exist():
        trials = loader.load_trials()
        print(f"\n✅ Data loader ready! {len(trials)} trials loaded.")
    else:
        print("\n❌ Files missing. Place your NPZ files in ./data/raw/")