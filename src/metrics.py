import numpy as np
import torch
import torch.nn.functional as F


def compute_dice_score(pred_binary, target_binary, smooth=1e-5):
    """
    Computes Dice Similarity Coefficient (DSC) between two binary masks.

    If both prediction and ground truth are empty, Dice = 1.0.
    If ground truth is empty but prediction is not, Dice = 0.0.
    """
    if isinstance(pred_binary, torch.Tensor):
        pred_binary = pred_binary.detach().cpu().numpy()
    if isinstance(target_binary, torch.Tensor):
        target_binary = target_binary.detach().cpu().numpy()

    pred_bool = pred_binary.astype(bool)
    target_bool = target_binary.astype(bool)

    pred_sum = np.sum(pred_bool)
    target_sum = np.sum(target_bool)

    # Empty mask corner cases
    if pred_sum == 0 and target_sum == 0:
        return 1.0
    if target_sum == 0 and pred_sum > 0:
        return 0.0

    intersection = np.sum(pred_bool & target_bool)
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    return float(dice)


def compute_sensitivity_specificity(pred_binary, target_binary):
    """
    Computes Sensitivity (Recall / True Positive Rate) and Specificity (True Negative Rate).
    """
    if isinstance(pred_binary, torch.Tensor):
        pred_binary = pred_binary.detach().cpu().numpy()
    if isinstance(target_binary, torch.Tensor):
        target_binary = target_binary.detach().cpu().numpy()

    p = pred_binary.astype(bool)
    t = target_binary.astype(bool)

    tp = np.sum(p & t)
    fp = np.sum(p & (~t))
    fn = np.sum((~p) & t)
    tn = np.sum((~p) & (~t))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0

    return float(sensitivity), float(specificity)


def compute_hd95(pred_binary, target_binary, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Computes 95th Percentile Hausdorff Distance (HD95).
    Uses scipy distance transform if available.
    """
    if isinstance(pred_binary, torch.Tensor):
        pred_binary = pred_binary.detach().cpu().numpy()
    if isinstance(target_binary, torch.Tensor):
        target_binary = target_binary.detach().cpu().numpy()

    p = pred_binary.astype(bool)
    t = target_binary.astype(bool)

    if not np.any(p) and not np.any(t):
        return 0.0
    if not np.any(p) or not np.any(t):
        return 373.13  # Standard BraTS maximum penalty distance

    try:
        from scipy.ndimage import distance_transform_edt, binary_erosion

        # Extract surface boundaries
        p_border = p ^ binary_erosion(p)
        t_border = t ^ binary_erosion(t)

        dt_target = distance_transform_edt(~t_border, sampling=voxel_spacing)
        dt_pred = distance_transform_edt(~p_border, sampling=voxel_spacing)

        d_p_to_t = dt_target[p_border]
        d_t_to_p = dt_pred[t_border]

        if len(d_p_to_t) == 0 or len(d_t_to_p) == 0:
            return 0.0

        all_distances = np.concatenate([d_p_to_t, d_t_to_p])
        return float(np.percentile(all_distances, 95))
    except Exception:
        return 0.0


def extract_brats_regions(mask):
    """
    Extracts BraTS evaluation sub-regions:
    - Whole Tumor (WT): labels 1, 2, 3 (or 1, 2, 4 in raw)
    - Tumor Core (TC): labels 1, 3 (or 1, 4 in raw)
    - Enhancing Tumor (ET): label 3 (or 4 in raw)

    Returns:
        tuple: (wt_mask, tc_mask, et_mask) as boolean arrays
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # Supports both converted labels (0, 1, 2, 3) and raw BraTS labels (0, 1, 2, 4)
    et = (mask == 3) | (mask == 4)
    tc = (mask == 1) | et
    wt = tc | (mask == 2)

    return wt, tc, et


def compute_brats_dice(pred, target):
    """
    Computes Dice score for the 3 BraTS regions: WT, TC, ET, and their Mean.

    pred: predicted label map [D, H, W] or logits [C, D, H, W] or [B, C, D, H, W]
    target: ground truth label map [D, H, W] or [B, D, H, W]

    Returns:
        dict: {'wt_dice': float, 'tc_dice': float, 'et_dice': float, 'mean_dice': float}
    """
    if isinstance(pred, torch.Tensor):
        if pred.ndim == 5:  # [B, C, D, H, W]
            pred = torch.argmax(pred, dim=1)
        elif pred.ndim == 4 and pred.shape[0] in [3, 4]:  # [C, D, H, W]
            pred = torch.argmax(pred, dim=0)
        pred = pred.detach().cpu().numpy()

    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    if pred.ndim > 3:
        pred = pred.squeeze()
    if target.ndim > 3:
        target = target.squeeze()

    pred_wt, pred_tc, pred_et = extract_brats_regions(pred)
    target_wt, target_tc, target_et = extract_brats_regions(target)

    wt_dice = compute_dice_score(pred_wt, target_wt)
    tc_dice = compute_dice_score(pred_tc, target_tc)
    et_dice = compute_dice_score(pred_et, target_et)
    mean_dice = (wt_dice + tc_dice + et_dice) / 3.0

    return {
        "wt_dice": wt_dice,
        "tc_dice": tc_dice,
        "et_dice": et_dice,
        "mean_dice": mean_dice
    }


class BraTSMetricAggregator:
    """
    Aggregates and summarizes BraTS metrics across multiple subjects/batches.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.records = {
            "wt_dice": [],
            "tc_dice": [],
            "et_dice": [],
            "mean_dice": [],
            "wt_sens": [],
            "tc_sens": [],
            "et_sens": [],
            "wt_spec": [],
            "tc_spec": [],
            "et_spec": [],
        }

    def update(self, pred, target, compute_sens_spec=False):
        dice_dict = compute_brats_dice(pred, target)
        self.records["wt_dice"].append(dice_dict["wt_dice"])
        self.records["tc_dice"].append(dice_dict["tc_dice"])
        self.records["et_dice"].append(dice_dict["et_dice"])
        self.records["mean_dice"].append(dice_dict["mean_dice"])

        if compute_sens_spec:
            if isinstance(pred, torch.Tensor) and pred.ndim in [4, 5]:
                pred_label = torch.argmax(pred, dim=1 if pred.ndim == 5 else 0).cpu().numpy().squeeze()
            else:
                pred_label = np.squeeze(pred)
            target_label = np.squeeze(target.cpu().numpy() if isinstance(target, torch.Tensor) else target)

            p_wt, p_tc, p_et = extract_brats_regions(pred_label)
            t_wt, t_tc, t_et = extract_brats_regions(target_label)

            w_sens, w_spec = compute_sensitivity_specificity(p_wt, t_wt)
            c_sens, c_spec = compute_sensitivity_specificity(p_tc, t_tc)
            e_sens, e_spec = compute_sensitivity_specificity(p_et, t_et)

            self.records["wt_sens"].append(w_sens)
            self.records["tc_sens"].append(c_sens)
            self.records["et_sens"].append(e_sens)
            self.records["wt_spec"].append(w_spec)
            self.records["tc_spec"].append(c_spec)
            self.records["et_spec"].append(e_spec)

        return dice_dict

    def summary(self):
        result = {}
        for key, values in self.records.items():
            if len(values) > 0:
                result[f"{key}_mean"] = float(np.mean(values))
                result[f"{key}_std"] = float(np.std(values))
                result[f"{key}_median"] = float(np.median(values))
        return result

    def summary_table(self):
        s = self.summary()
        lines = [
            "+----------------------+---------+---------+---------+",
            "| Region               | Mean    | Std     | Median  |",
            "+----------------------+---------+---------+---------+",
            f"| Whole Tumor (WT)     | {s.get('wt_dice_mean', 0.0):.4f}  | {s.get('wt_dice_std', 0.0):.4f}  | {s.get('wt_dice_median', 0.0):.4f}  |",
            f"| Tumor Core (TC)      | {s.get('tc_dice_mean', 0.0):.4f}  | {s.get('tc_dice_std', 0.0):.4f}  | {s.get('tc_dice_median', 0.0):.4f}  |",
            f"| Enhancing Tumor (ET) | {s.get('et_dice_mean', 0.0):.4f}  | {s.get('et_dice_std', 0.0):.4f}  | {s.get('et_dice_median', 0.0):.4f}  |",
            f"| Average Mean Dice    | {s.get('mean_dice_mean', 0.0):.4f}  | {s.get('mean_dice_std', 0.0):.4f}  | {s.get('mean_dice_median', 0.0):.4f}  |",
            "+----------------------+---------+---------+---------+",
        ]
        return "\n".join(lines)
