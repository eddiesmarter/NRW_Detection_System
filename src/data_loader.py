import pandas as pd

def load_utility_data(file_path):
    """Loads raw CSV data and performs initial cleaning for missing values."""
    df = pd.read_csv(file_path)
    # Handling missing values as per proposal limitations
    df = df.fillna(df.median(numeric_only=True))
    return df
