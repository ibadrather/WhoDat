from typing import Any, Optional, Tuple
import pandas as pd
import torch
import cv2
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import numpy as np


class WhoDatDataset(Dataset):
    """
    A custom dataset class for loading and preprocessing the WhoDat dataset.

    Args:
        dataframe (pd.DataFrame): A pandas DataFrame containing the dataset information,
                                  with columns 'file_path' and 'class'.
        data_augmentation (Optional[Any]): A data augmentation object from Albumentations
                                          library or any other library with a similar API.
                                          Default is None.

    Attributes:
        data (np.ndarray): An array containing the file paths of the images.
        _class (np.ndarray): An array containing the encoded class labels of the images.
        label_encoder (LabelEncoder): A scikit-learn LabelEncoder object used for encoding
                                      and decoding the class labels.
        data_augmentation (Optional[Any]): A data augmentation object.
    """

    def __init__(
        self, dataframe: pd.DataFrame, data_augmentation: Optional[Any] = None
    ):
        self.data = dataframe["file_path"].values
        self._class = dataframe["class"].values

        self.label_encoder = LabelEncoder()
        self._class = self.label_encoder.fit_transform(self._class)

        self.data_augmentation = data_augmentation

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data)

    def decode_labels(self, labels: torch.Tensor) -> np.ndarray:
        """
        Decodes the encoded labels using the label encoder.

        Args:
            labels (torch.Tensor): A tensor containing the encoded labels.

        Returns:
            np.ndarray: An array containing the decoded labels.

        Example:
            original_labels = dataset.decode_labels(encoded_labels)
        """
        return self.label_encoder.inverse_transform(labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads and returns the image and its corresponding class label at the specified index.

        Args:
            index (int): The index of the sample to load.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the image and its corresponding class label.
        """
        # Get the image
        image_dir = self.data[index]

        # Read the image
        image = cv2.imread(image_dir, cv2.IMREAD_COLOR)

        # Get the class
        _class = self._class[index]

        # Apply data augmentation
        if self.data_augmentation:
            image = self.data_augmentation(image=image)["image"]

        # Convert the image to a tensor
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        # Convert the class to a tensor
        _class = torch.tensor(_class, dtype=torch.long)

        return image, _class


def main():
    # Get the absolute path of the parent directory of the script
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Add the parent directory to sys.path
    sys.path.append(parent_dir)

    from Common.DataAugmentation import AlbumentationsDataAugmentation

    # Load the data
    data = pd.read_csv("/home/ibad/Desktop/WhoDat/WhoDat_Faces/train.csv")

    # Data augmentation
    data_augmentation = AlbumentationsDataAugmentation(image_size=(128, 128))

    # Create the dataset
    dataset = WhoDatDataset(data, data_augmentation=data_augmentation)

    # Get the first image
    image, _class = dataset[0]

    print(image.shape)
    print(_class)


if __name__ == "__main__":

    import pandas as pd
    import os
    import sys

    os.system("clear")
    main()
