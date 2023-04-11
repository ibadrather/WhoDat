import pandas as pd
import numpy as np
import os
import sys
import pytest
# Get the absolute path of the parent directory of the script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Add the parent directory to sys.path
sys.path.append(parent_dir)

from Model_Dev.utils import balance_dataframe_by_resampling_class

def test_balance_dataframe_by_resampling_class():
    # Create a sample imbalanced dataframe
    data = {'class': [0]*100 + [1]*10,
            'feature1': np.random.rand(110),
            'feature2': np.random.rand(110)}
    df = pd.DataFrame(data)

    # Test oversampling
    balanced_df = balance_dataframe_by_resampling_class(df, 'class', 100, replace=True)
    assert len(balanced_df) == 200, "Oversampling failed: incorrect length"
    assert balanced_df['class'].value_counts().iloc[0] == 100, "Oversampling failed: incorrect count for class 0"
    assert balanced_df['class'].value_counts().iloc[1] == 100, "Oversampling failed: incorrect count for class 1"

    # Test undersampling
    balanced_df = balance_dataframe_by_resampling_class(df, 'class', 10, replace=False)
    assert len(balanced_df) == 20, "Undersampling failed: incorrect length"
    assert balanced_df['class'].value_counts().iloc[0] == 10, "Undersampling failed: incorrect count for class 0"
    assert balanced_df['class'].value_counts().iloc[1] == 10, "Undersampling failed: incorrect count for class 1"

    # Test edge case: when sample_size is larger than the number of samples in any class and replace=False
    with pytest.raises(ValueError):
        balance_dataframe_by_resampling_class(df, 'class', 200, replace=False)
