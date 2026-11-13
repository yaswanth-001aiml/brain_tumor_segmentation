import os
import glob
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class BraTSDataset(Dataset):

    def __init__(self, dataset_dir, verbose=False):

        self.dataset_dir = dataset_dir
        self.verbose = verbose

        self.patient_dirs = sorted(
            [
                os.path.join(dataset_dir, d)
                for d in os.listdir(dataset_dir)
                if os.path.isdir(os.path.join(dataset_dir, d))
            ]
        )

        if len(self.patient_dirs) == 0:
            raise RuntimeError(
                f"No patient folders found in: {dataset_dir}"
            )

        print(f"Found {len(self.patient_dirs)} patients")

    def load_nifti(self, path):

        nii = nib.load(path)

        data = nii.get_fdata()

        return data.astype(np.float32)

    def find_file(self, patient_dir, suffix):

        # Exact modality suffix
        pattern = os.path.join(
            patient_dir,
            f"*{suffix}.nii.gz"
        )

        files = glob.glob(pattern)

        if len(files) == 0:
            raise FileNotFoundError(
                f"Could not find {suffix} in {patient_dir}"
            )

        return files[0]

    def __getitem__(self, index):

        patient_dir = self.patient_dirs[index]

        if self.verbose:
            print(f"\nLoading patient: {os.path.basename(patient_dir)}")

        # -------------------------
        # Load MRI modalities
        # -------------------------

        t1_path = self.find_file(
            patient_dir,
            "_t1"
        )

        t1ce_path = self.find_file(
            patient_dir,
            "_t1ce"
        )

        t2_path = self.find_file(
            patient_dir,
            "_t2"
        )

        flair_path = self.find_file(
            patient_dir,
            "_flair"
        )

        seg_path = self.find_file(
            patient_dir,
            "_seg"
        )

        if self.verbose:
            print("T1    :", os.path.basename(t1_path))
            print("T1ce  :", os.path.basename(t1ce_path))
            print("T2    :", os.path.basename(t2_path))
            print("FLAIR :", os.path.basename(flair_path))
            print("SEG   :", os.path.basename(seg_path))

        # -------------------------
        # Read NIfTI volumes
        # -------------------------

        t1 = self.load_nifti(t1_path)
        t1ce = self.load_nifti(t1ce_path)
        t2 = self.load_nifti(t2_path)
        flair = self.load_nifti(flair_path)

        seg = self.load_nifti(seg_path)

        # -------------------------
        # Stack four modalities
        # -------------------------

        image = np.stack(
            [
                t1,
                t1ce,
                t2,
                flair
            ],
            axis=0
        )

        # -------------------------
        # Convert to tensors
        # -------------------------

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        seg = torch.tensor(
            seg,
            dtype=torch.long
        )

        return image, seg

    def __len__(self):

        return len(self.patient_dirs)