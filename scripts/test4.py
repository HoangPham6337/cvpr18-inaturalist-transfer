import numpy as np

train_features = np.load("../feature/inception_v3_iNat_299/haute_garonne_feature_train.npy")
train_labels = np.load("../feature/inception_v3_iNat_299/haute_garonne_label_train.npy")
val_features = np.load("../feature/inception_v3_iNat_299/haute_garonne_feature_val.npy")
val_labels = np.load("../feature/inception_v3_iNat_299/haute_garonne_label_val.npy")

num_train_samples = train_features.shape[0]
num_val_samples = val_features.shape[0]

num_classes = len(set(train_labels))  # Assumes labels are categorical integers

haute_garonne_dataset = {
    'num_samples': {'train': num_train_samples, 'validation': num_val_samples},
    'num_classes': num_classes
}

print(f"'haute_garonne': {haute_garonne_dataset},")