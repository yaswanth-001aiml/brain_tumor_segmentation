from collections import OrderedDict
import random

import torch
from torch.utils.data import Dataset

from .preprocessing import (
    preprocess_mri,
    convert_mask
)


class BraTSPatchDataset(Dataset):
    """
    3D Patch-based Dataset for BraTS MRI volumes.

    Features:
    - In-memory volume caching with bounding-box cropping to eliminate disk I/O.
    - Balanced patch sampling: ~67% tumor-focused, ~33% healthy brain tissue.
    - 3D spatial augmentations (RandomFlip3D, RandomIntensityShift).
    """

    def __init__(
        self,
        base_dataset,
        patch_size=64,
        patches_per_patient=4,
        transform=None,
        tumor_prob=0.67,
        cache_data=True
    ):
        self.base_dataset = base_dataset
        self.patch_size = patch_size
        self.patches_per_patient = patches_per_patient
        self.transform = transform
        self.tumor_prob = tumor_prob
        self.cache_data = cache_data

        self.cached_samples = []
        if self.cache_data:
            self._preload_and_cache()

    def _preload_and_cache(self):
        print(f"Caching and cropping {len(self.base_dataset)} volumes into memory...", flush=True)
        ps = self.patch_size

        for i in range(len(self.base_dataset)):
            raw_image, raw_mask = self.base_dataset[i]

            norm_image = preprocess_mri(raw_image)
            c_mask = convert_mask(raw_mask)

            # Brain tissue mask (any modality non-zero)
            brain_mask = (
                (norm_image[0] != 0) |
                (norm_image[1] != 0) |
                (norm_image[2] != 0) |
                (norm_image[3] != 0)
            )

            idx = torch.nonzero(brain_mask)
            if len(idx) > 0:
                min_d = max(0, idx[:, 0].min().item() - 4)
                max_d = min(norm_image.shape[1], idx[:, 0].max().item() + 4)
                min_h = max(0, idx[:, 1].min().item() - 4)
                max_h = min(norm_image.shape[2], idx[:, 1].max().item() + 4)
                min_w = max(0, idx[:, 2].min().item() - 4)
                max_w = min(norm_image.shape[3], idx[:, 2].max().item() + 4)
            else:
                min_d, max_d = 0, norm_image.shape[1]
                min_h, max_h = 0, norm_image.shape[2]
                min_w, max_w = 0, norm_image.shape[3]

            # Ensure cropped dimensions are at least patch_size
            if (max_d - min_d) < ps:
                diff = ps - (max_d - min_d)
                min_d = max(0, min_d - diff // 2)
                max_d = min(norm_image.shape[1], max_d + (diff - diff // 2))
            if (max_h - min_h) < ps:
                diff = ps - (max_h - min_h)
                min_h = max(0, min_h - diff // 2)
                max_h = min(norm_image.shape[2], max_h + (diff - diff // 2))
            if (max_w - min_w) < ps:
                diff = ps - (max_w - min_w)
                min_w = max(0, min_w - diff // 2)
                max_w = min(norm_image.shape[3], max_w + (diff - diff // 2))

            crop_image = norm_image[:, min_d:max_d, min_h:max_h, min_w:max_w]
            crop_mask = c_mask[min_d:max_d, min_h:max_h, min_w:max_w]

            # Cache tumor and brain locations in cropped space
            tumor_locs = torch.nonzero(crop_mask > 0)
            crop_brain = (crop_image[0] != 0) | (crop_image[1] != 0) | (crop_image[2] != 0) | (crop_image[3] != 0)
            brain_locs = torch.nonzero(crop_brain)

            self.cached_samples.append({
                "image": crop_image,
                "mask": crop_mask,
                "tumor_locs": tumor_locs,
                "brain_locs": brain_locs
            })

        print(f"Successfully cached {len(self.cached_samples)} volumes in memory.", flush=True)

    def __len__(self):
        return len(self.base_dataset) * self.patches_per_patient

    def __getitem__(self, index):
        patient_idx = index // self.patches_per_patient
        ps = self.patch_size

        if self.cache_data:
            sample = self.cached_samples[patient_idx]
            image = sample["image"]
            mask = sample["mask"]
            tumor_locs = sample["tumor_locs"]
            brain_locs = sample["brain_locs"]
        else:
            if not hasattr(self, '_lru_cache'):
                self._lru_cache = OrderedDict()
                self._lru_max = 2

            if patient_idx in self._lru_cache:
                image, mask, tumor_locs, brain_locs = self._lru_cache[patient_idx]
                self._lru_cache.move_to_end(patient_idx)
            else:
                raw_image, raw_mask = self.base_dataset[patient_idx]
                image = preprocess_mri(raw_image)
                mask = convert_mask(raw_mask)

                # Pre-crop to bounding box to reduce memory from ~143MB to ~30MB
                brain_mask = (image[0] != 0) | (image[1] != 0) | (image[2] != 0) | (image[3] != 0)
                idx = torch.nonzero(brain_mask)
                if len(idx) > 0:
                    min_d = max(0, idx[:, 0].min().item() - 2)
                    max_d = min(image.shape[1], idx[:, 0].max().item() + 2)
                    min_h = max(0, idx[:, 1].min().item() - 2)
                    max_h = min(image.shape[2], idx[:, 1].max().item() + 2)
                    min_w = max(0, idx[:, 2].min().item() - 2)
                    max_w = min(image.shape[3], idx[:, 2].max().item() + 2)

                    if max_d - min_d < ps:
                        max_d = min(image.shape[1], min_d + ps)
                    if max_h - min_h < ps:
                        max_h = min(image.shape[2], min_h + ps)
                    if max_w - min_w < ps:
                        max_w = min(image.shape[3], min_w + ps)

                    image = image[:, min_d:max_d, min_h:max_h, min_w:max_w].clone()
                    mask = mask[min_d:max_d, min_h:max_h, min_w:max_w].clone()

                tumor_locs = torch.nonzero(mask > 0)
                brain_locs = torch.nonzero(
                    (image[0] != 0) | (image[1] != 0) | (image[2] != 0) | (image[3] != 0)
                )

                while len(self._lru_cache) >= self._lru_max:
                    _, old_item = self._lru_cache.popitem(last=False)
                    del old_item

                self._lru_cache[patient_idx] = (image, mask, tumor_locs, brain_locs)

        _, depth, height, width = image.shape

        # Balanced sampling: tumor vs brain tissue
        sample_tumor = (random.random() < self.tumor_prob) and (len(tumor_locs) > 0)

        if sample_tumor:
            idx = random.randint(0, len(tumor_locs) - 1)
            center = tumor_locs[idx]
            center_d, center_h, center_w = int(center[0]), int(center[1]), int(center[2])
        elif len(brain_locs) > 0:
            idx = random.randint(0, len(brain_locs) - 1)
            center = brain_locs[idx]
            center_d, center_h, center_w = int(center[0]), int(center[1]), int(center[2])
        else:
            center_d = random.randint(0, depth - 1)
            center_h = random.randint(0, height - 1)
            center_w = random.randint(0, width - 1)

        # Calculate bounding box starts
        start_d = max(0, min(center_d - ps // 2, depth - ps))
        start_h = max(0, min(center_h - ps // 2, height - ps))
        start_w = max(0, min(center_w - ps // 2, width - ps))

        image_patch = image[:, start_d:start_d + ps, start_h:start_h + ps, start_w:start_w + ps]
        mask_patch = mask[start_d:start_d + ps, start_h:start_h + ps, start_w:start_w + ps]

        # Handle edge padding if patch is smaller than required ps
        if (image_patch.shape[1] < ps or image_patch.shape[2] < ps or image_patch.shape[3] < ps):
            pad_d = max(0, ps - image_patch.shape[1])
            pad_h = max(0, ps - image_patch.shape[2])
            pad_w = max(0, ps - image_patch.shape[3])
            image_patch = torch.nn.functional.pad(image_patch, (0, pad_w, 0, pad_h, 0, pad_d))
            mask_patch = torch.nn.functional.pad(mask_patch, (0, pad_w, 0, pad_h, 0, pad_d))

        if self.transform is not None:
            image_patch, mask_patch = self.transform(image_patch, mask_patch)

        return image_patch, mask_patch
