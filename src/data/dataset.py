import os
import glob

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class BraTSDataset(Dataset):

    def __init__(
        self,
        patient_dirs,
        training=True
    ):

        self.patient_dirs = patient_dirs

        self.training = training

        if len(patient_dirs) == 0:

            raise RuntimeError(
                "No patient directories provided."
            )

    def load_nifti(self, path):

        nii = nib.load(path)

        data = nii.get_fdata()

        return data.astype(
            np.float32
        )

    def find_file(
        self,
        patient_dir,
        suffixes
    ):

        if isinstance(suffixes, str):
            suffixes = [suffixes]

        for suffix in suffixes:
            pattern = os.path.join(
                patient_dir,
                f"*{suffix}.nii.gz"
            )
            files = glob.glob(pattern)
            if len(files) > 0:
                return files[0]

        raise FileNotFoundError(
            f"Missing any of {suffixes} in "
            f"{patient_dir}"
        )

    def load_patient(
        self,
        patient_dir
    ):

        # --------------------------------
        # MRI modalities (supports 2021 [_t1, etc.] and 2023 [-t1n, etc.])
        # --------------------------------

        t1 = self.load_nifti(
            self.find_file(
                patient_dir,
                ["_t1", "-t1n"]
            )
        )

        t1ce = self.load_nifti(
            self.find_file(
                patient_dir,
                ["_t1ce", "-t1c"]
            )
        )

        t2 = self.load_nifti(
            self.find_file(
                patient_dir,
                ["_t2", "-t2w"]
            )
        )

        flair = self.load_nifti(
            self.find_file(
                patient_dir,
                ["_flair", "-t2f"]
            )
        )

        # --------------------------------
        # Ground truth
        # --------------------------------

        seg = self.load_nifti(
            self.find_file(
                patient_dir,
                ["_seg", "-seg"]
            )
        )

        # --------------------------------
        # Stack modalities
        # --------------------------------

        image = np.stack(
            [
                t1,
                t1ce,
                t2,
                flair
            ],
            axis=0
        )

        return image, seg

    def __getitem__(self, index):

        patient_dir = self.patient_dirs[
            index
        ]

        image, mask = self.load_patient(
            patient_dir
        )

        return image, mask

    def __len__(self):

        return len(
            self.patient_dirs
        )
