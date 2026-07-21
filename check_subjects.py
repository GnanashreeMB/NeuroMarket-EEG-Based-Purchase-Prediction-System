# check_subjects.py
import numpy as np

# Load your subjects array
subjects = np.load('data/raw/SubjectsNoImage.npz')['arr_0']

# Get unique subjects
unique_subjects = np.unique(subjects)
print(f"Total trials: {len(subjects)}")
print(f"Unique subjects: {len(unique_subjects)}")
print(f"Subject IDs: {unique_subjects}")