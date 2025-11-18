import pandas as pd

from src.data_loading import load_raw_data


def test_load_raw_data_returns_dataframe() -> None:
    """load_raw_data should return a non-empty pandas DataFrame."""
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
