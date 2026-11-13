import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSupervisionHead3D(nn.Module):
    """
    Segmentation head used for deep supervision.

    Converts an intermediate decoder feature map into
    segmentation logits.
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.head = nn.Conv3d(
            in_channels,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):
        return self.head(x)


class DeepSupervisionLossWrapper:
    """
    Utility for handling multiple segmentation outputs.

    The outputs can come from different decoder resolutions.

    Each output is compared with a target resized to the
    corresponding spatial resolution.
    """

    def __init__(
        self,
        weights=None
    ):
        self.weights = weights

    def prepare_target(
        self,
        target,
        output
    ):
        """
        Resize segmentation target to match the output.

        target:
            [B, D, H, W]

        output:
            [B, C, D, H, W]
        """

        if target.ndim != 4:
            raise ValueError(
                "Target must have shape [B, D, H, W]."
            )

        target = target.unsqueeze(1).float()

        target = F.interpolate(
            target,
            size=output.shape[2:],
            mode="nearest"
        )

        target = target.squeeze(1).long()

        return target

    def normalize_weights(
        self,
        number_of_outputs
    ):
        """
        Normalize deep-supervision weights so that
        their sum is 1.
        """

        if self.weights is None:

            weights = [
                1.0 / number_of_outputs
                for _ in range(number_of_outputs)
            ]

        else:

            if len(self.weights) != number_of_outputs:
                raise ValueError(
                    "Number of weights must match "
                    "number of outputs."
                )

            total = sum(self.weights)

            if total <= 0:
                raise ValueError(
                    "Deep supervision weights must "
                    "have a positive sum."
                )

            weights = [
                weight / total
                for weight in self.weights
            ]

        return weights

    def __call__(
        self,
        outputs
    ):
        """
        Prepare outputs and corresponding weights.

        Returns:
            list of (output, weight)
        """

        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        weights = self.normalize_weights(
            len(outputs)
        )

        return [
            (output, weight)
            for output, weight in zip(
                outputs,
                weights
            )
        ]