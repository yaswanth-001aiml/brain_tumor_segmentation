import random

import torch


class RandomFlip3D:

    def __init__(
        self,
        probability=0.5
    ):

        self.probability = probability

    def __call__(
        self,
        image,
        mask
    ):

        # Depth
        if random.random() < self.probability:

            image = torch.flip(
                image,
                dims=[1]
            )

            mask = torch.flip(
                mask,
                dims=[0]
            )

        # Height
        if random.random() < self.probability:

            image = torch.flip(
                image,
                dims=[2]
            )

            mask = torch.flip(
                mask,
                dims=[1]
            )

        # Width
        if random.random() < self.probability:

            image = torch.flip(
                image,
                dims=[3]
            )

            mask = torch.flip(
                mask,
                dims=[2]
            )

        return image, mask


class RandomIntensityShift:

    def __init__(
        self,
        probability=0.3,
        max_shift=0.1
    ):

        self.probability = probability

        self.max_shift = max_shift

    def __call__(
        self,
        image,
        mask
    ):

        if random.random() < self.probability:

            shift = (
                random.uniform(
                    -self.max_shift,
                    self.max_shift
                )
            )

            image = image + shift

        return image, mask


class Compose3D:

    def __init__(
        self,
        transforms
    ):

        self.transforms = transforms

    def __call__(
        self,
        image,
        mask
    ):

        for transform in self.transforms:

            image, mask = transform(
                image,
                mask
            )

        return image, mask