import numpy as np

# Load labels
labels = np.load('data/raw/LabelsNoImage.npz')['arr_0']

print(f"Unique label values: {np.unique(labels)}")
print(f"Value counts:")
unique, counts = np.unique(labels, return_counts=True)
for val, count in zip(unique, counts):
    print(f"  {val}: {count} trials")