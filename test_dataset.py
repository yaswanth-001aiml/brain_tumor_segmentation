import sys
import os


sys.path.append(
    os.path.join(os.path.dirname(__file__), "src")
)

from dataset import BraTSDataset
from preprocessing import preprocess_mri

# Change this if your dataset is somewhere else
DATASET_PATH = "dataset/BraTS2021_Training_Data/BraTS2021_Training_Data"

dataset = BraTSDataset(DATASET_PATH)

print("\nDataset verification")
print("--------------------")

print("Number of patients:", len(dataset))

image, mask = dataset[0]
# Preprocess MRI
processed_image = preprocess_mri(image)

print("\nAfter preprocessing")
print("-------------------")

print("Processed shape:",
      processed_image.shape)

print("Processed minimum:",
      processed_image.min().item())

print("Processed maximum:",
      processed_image.max().item())

print("Processed mean:",
      processed_image.mean().item())

print("Processed std:",
      processed_image.std().item())

print("MRI shape :", image.shape)
print("Mask shape:", mask.shape)

print("\nMRI datatype :", image.dtype)
print("Mask datatype:", mask.dtype)

print("\nMRI minimum:", image.min().item())
print("MRI maximum:", image.max().item())

print("\nMask labels:", mask.unique().tolist())