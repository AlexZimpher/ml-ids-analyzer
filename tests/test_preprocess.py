import pandas as pd
import numpy as np
from preprocess import clean_and_label

def test_clean_and_label_basic():
    df = pd.DataFrame({
        'A': [1, 2, np.nan],
        'B': [1, 1, 1],
        'Label': ['BENIGN', 'Attack', 'BENIGN']
    })
    cleaned = clean_and_label(df)
    assert 'Label' in cleaned.columns
    assert set(cleaned['Label'].unique()) <= {0, 1}
