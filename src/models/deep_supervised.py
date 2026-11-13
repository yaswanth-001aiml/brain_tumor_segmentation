import torch
import torch.nn as nn


class DeepSupervisionMixin:
    """
    Common helper for models that expose intermediate decoder
    features for deep supervision.
    """

    def create_deep_supervision_outputs(
        self,
        decoder_features,
        heads,
    ):
        """
        Convert decoder features into segmentation logits.

        decoder_features:
            List ordered from highest-resolution decoder feature
            to lowest-resolution decoder feature.

        heads:
            Corresponding 1x1x1 segmentation heads.

        Returns:
            List of segmentation logits.
        """

        if len(decoder_features) != len(heads):
            raise ValueError(
                "Number of decoder features must match "
                "number of deep-supervision heads."
            )

        outputs = []

        for feature, head in zip(
            decoder_features,
            heads
        ):
            outputs.append(
                head(feature)
            )

        return outputs