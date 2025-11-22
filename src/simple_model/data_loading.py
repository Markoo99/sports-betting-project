from pathlib import Path
from typing import Union

import pandas as pd


# Default path to your odds data
DATA_PATH = Path("data") / "oddsData.csv"


def load_raw_data(path: Union[str, Path] = DATA_PATH) -> pd.DataFrame:
    """
    Load the raw NBA odds dataset from a CSV file.

    Parameters
    ----------
    path : str or Path, optional
        Path to the CSV file. Defaults to DATA_PATH ("data/oddsData.csv").

    Returns
    -------
    pd.DataFrame
        Loaded data as a pandas DataFrame.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    ValueError
        If the loaded DataFrame is empty.
    """
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(f"Loaded DataFrame is empty: {csv_path}")

    return df


if __name__ == "__main__":
    # Simple manual check: run this file to see basic info about the data
    df = load_raw_data()
    print("✅ Data loaded successfully")
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())

