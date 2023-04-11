import pytest
import numpy as np
import os
import sys

# Get the absolute path of the parent directory of the script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add the parent directory to sys.path
sys.path.append(parent_dir)

from Common.DataAugmentation import AlbumentationsDataAugmentation


@pytest.fixture
def sample_image():
    return np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)


def test_data_augmentation(sample_image):
    augmentations = {
        "horizontal_flip": True,
        "vertical_flip": True,
        "random_rotate_90": False,
        "transpose": True,
        "medium_augmentations": True,
        "clahe": True,
        "random_brightness_contrast": True,
        "random_gamma": True,
    }
    data_augmentation = AlbumentationsDataAugmentation(
        image_size=(224, 224), options=augmentations
    )

    # Apply the data augmentation
    augmented_image = data_augmentation(image=sample_image)

    # Check if the augmented image has the correct shape
    assert augmented_image["image"].shape == (224, 224, 3)

    # Check if the augmented image is different from the original image
    assert not np.array_equal(sample_image, augmented_image["image"])
