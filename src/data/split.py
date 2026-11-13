import os
import random


def get_patient_directories(dataset_dir):

    patients = sorted(
        [
            os.path.join(dataset_dir, name)
            for name in os.listdir(dataset_dir)
            if os.path.isdir(
                os.path.join(dataset_dir, name)
            )
        ]
    )

    if len(patients) == 0:
        raise RuntimeError(
            f"No patient folders found in {dataset_dir}"
        )

    return patients


def split_dataset(
    dataset_dir,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):

    if abs(
        train_ratio +
        val_ratio +
        test_ratio -
        1.0
    ) > 1e-6:

        raise ValueError(
            "Split ratios must sum to 1."
        )

    patients = get_patient_directories(
        dataset_dir
    )

    # Reproducible split
    random_generator = random.Random(seed)

    random_generator.shuffle(patients)

    total = len(patients)

    train_end = int(
        total * train_ratio
    )

    val_end = train_end + int(
        total * val_ratio
    )

    train_patients = patients[
        :train_end
    ]

    val_patients = patients[
        train_end:val_end
    ]

    test_patients = patients[
        val_end:
    ]

    return (
        train_patients,
        val_patients,
        test_patients
    )