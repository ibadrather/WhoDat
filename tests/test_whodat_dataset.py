import os
import sys

# Get the absolute path of the parent directory of the script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add the parent directory to sys.path
sys.path.append(parent_dir)

import pytest
import pandas as pd
import torch
from Model_Dev.dataloading import WhoDatDataset


@pytest.fixture
def test_dataframe() -> pd.DataFrame:
    df = pd.read_csv("WhoDat_Faces/test.csv")
    return df


@pytest.fixture
def whodat_dataset(test_dataframe: pd.DataFrame) -> WhoDatDataset:
    return WhoDatDataset(test_dataframe)


def test_getitem(whodat_dataset: WhoDatDataset):
    image, label = whodat_dataset[0]
    assert isinstance(image, torch.Tensor), "Image should be a torch.Tensor"
    assert image.dtype == torch.float32, "Image should have dtype torch.float32"
    assert isinstance(label, torch.Tensor), "Label should be a torch.Tensor"
    assert label.dtype == torch.long, "Label should have dtype torch.long"

    with pytest.raises(IndexError):
        _ = whodat_dataset[0][3]
