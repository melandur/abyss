import os

import numpy as np

# Root training directory
root_dir = "/home/melandur/code/abyss/data/training/gbm"

# Define class label mapping
class_mapping = {"t1": 0, "t2": 1}
num_classes = len(class_mapping)


def create_labels(root_dir):
    for dirpath, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".nii.gz"):
                filepath = os.path.join(dirpath, file)

                # Extract modality: get part after last underscore, before ".nii.gz"
                modality = file.rsplit("_", 1)[-1].replace(".nii.gz", "").lower()

                if modality in class_mapping:
                    label_index = class_mapping[modality]
                    # Create one-hot vector
                    one_hot_vector = np.zeros(num_classes, dtype=int)
                    one_hot_vector[label_index] = 1

                    label_path = filepath.replace(".nii.gz", ".txt")
                    # Save one-hot vector as space-separated values
                    with open(label_path, "w") as f:
                        f.write(" ".join(map(str, one_hot_vector)))

                    print(f"✔ One-hot label {one_hot_vector} written to {label_path}")
                else:
                    print(f"⚠ Unknown modality in {filepath}, skipping...")


if __name__ == "__main__":
    create_labels(root_dir)
