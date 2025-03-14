import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

inat_features_train = np.load("../feature/inception_v3_iNat_299/inat2017_feature_train.npy")
inat_labels_train = np.load("../feature/inception_v3_iNat_299/inat2017_label_train.npy")
inat_features_val = np.load("../feature/inception_v3_iNat_299/inat2017_feature_val.npy")
inat_labels_val = np.load("../feature/inception_v3_iNat_299/inat2017_label_val.npy")

inat_combined_features = np.concatenate((inat_features_train, inat_features_val), axis=0)
inat_combined_labels = np.concatenate((inat_labels_train, inat_labels_val), axis=0)


hg_features_train = np.load("../feature/inception_v3_iNat_299/haute_garonne_feature_train.npy")
hg_labels_train = np.load("../feature/inception_v3_iNat_299/haute_garonne_label_train.npy")
hg_features_val = np.load("../feature/inception_v3_iNat_299/haute_garonne_feature_val.npy")
hg_labels_val = np.load("../feature/inception_v3_iNat_299/haute_garonne_label_val.npy")

hg_combined_features = np.concatenate((hg_features_train, hg_features_val), axis=0)
hg_combined_labels = np.concatenate((hg_labels_train, hg_labels_val), axis=0)
print("Checkpoint 1")


unique_hg_species = set(np.unique(hg_combined_labels))
hg_indices = [i for i, label in enumerate(inat_combined_labels) if label in unique_hg_species]

hg_full_features = inat_combined_features[hg_indices]
hg_full_labels = inat_combined_labels[hg_indices]


smote = SMOTE(sampling_strategy="auto", random_state=42, k_neighbors=min(5, len(hg_full_labels)-1))
hg_balanced_features, hg_balanced_labels = smote.fit_resample(hg_full_features, hg_full_labels)

print("Checkpoint 2")

other_indices = [i for i, label in enumerate(inat_combined_labels) if label not in unique_hg_species]
other_features = inat_combined_features[other_indices]
other_labels = np.full(len(other_features), fill_value=max(hg_combined_labels) + 1)

final_features = np.concatenate((hg_balanced_features, other_features), axis=0)
final_labels = np.concatenate((hg_balanced_labels, other_labels), axis=0)
print("Checkpoint 3")

hg_features_train_bal, hg_features_val_bal, hg_labels_train_bal, hg_labels_val_bal = train_test_split(
    final_features, final_labels, test_size=0.2, random_state=42, stratify=final_labels
)

print("Saving features")
np.save("./feature/haute_garonne_train_balanced.npy", hg_features_train_bal)
np.save("./feature/haute_garonne_train_labels_balanced.npy", hg_labels_train_bal)
np.save("./feature/haute_garonne_val_balanced.npy", hg_features_val_bal)
np.save("./feature/haute_garonne_val_labels_balanced.npy", hg_labels_val_bal)