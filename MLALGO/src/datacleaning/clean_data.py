import pandas as pd

def load_dataset(path):
    """Load the CSV dataset."""
    df = pd.read_csv(path)
    print(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def clean_dataset(df):
    """Simple cleaning: remove duplicates and missing values."""
    cleaned = df.drop_duplicates().dropna()
    print(f"✅ Cleaned dataset: {cleaned.shape[0]} rows remain after cleaning.")
    return cleaned
