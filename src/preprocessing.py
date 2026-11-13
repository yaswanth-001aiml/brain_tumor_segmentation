import numpy as np
import torch


def normalize_volume(volume):
    """
    In-place Z-score normalization of a 3D MRI volume slice.
    Avoids duplicate array allocations to keep RAM usage minimal.
    """
    if volume.dtype != np.float32:
        volume = volume.astype(np.float32)

    mask = volume > 0
    if not np.any(mask):
        return volume

    mean = float(volume[mask].mean())
    std = float(volume[mask].std())

    if std > 1e-8:
        volume[mask] = (volume[mask] - mean) / std

    return volume


def preprocess_mri(image):
    """
    Normalizes multi-modal MRI volume in-place to minimize RAM usage.
    image shape: [4, D, H, W]
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    if image.dtype != np.float32:
        image = image.astype(np.float32)

    for channel in range(image.shape[0]):
        image[channel] = normalize_volume(image[channel])

    return torch.from_numpy(image)


def convert_mask(mask):
    """
    BraTS 2021 & 2023 label converter:
    0 -> 0 Background
    1 -> 1 Necrotic Core (NCR/NET)
    2 -> 2 Edema (ED)
    3 -> 3 Enhancing Tumor (ET) [BraTS 2023 standard]
    4 -> 3 Enhancing Tumor (ET) [BraTS 2021 standard]
    """
    if isinstance(mask, np.ndarray):
        mask = torch.as_tensor(mask, dtype=torch.long)
    else:
        mask = mask.clone().long()

    converted = torch.zeros_like(mask)
    converted[mask == 1] = 1
    converted[mask == 2] = 2
    converted[mask == 3] = 3
    converted[mask == 4] = 3

    return converted
