"""import pandas as pd

def load_data(filename: str = 'Spotify_final_dataset.csv') -> pd.DataFrame:
    """Load the Spotify dataset from the working directory."""
    return pd.read_csv(filename, low_memory=False)

def find_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a summary table of missing values:
      – count of missing per column
      – percentage of missing per column
    """
    missing_count = df.isnull().sum()
    missing_pct   = 100 * missing_count / len(df)
    summary = pd.DataFrame({
        'missing_count': missing_count,
        'missing_pct': missing_pct
    }).sort_values(by='missing_count', ascending=False)
    return summary

def show_rows_with_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the subset of rows that have at least one missing value.
    """
    return df[df.isnull().any(axis=1)]

if __name__ == '__main__':
    # 1. Load
    df = load_data()

    # 2. Summary of missingness
    print("=== Missing Data Summary ===")
    print(find_missing_data(df), "\n")

    # 3. If you want to inspect the actual rows:
    missing_rows = show_rows_with_missing(df)
    print(f"Rows with ≥1 missing value: {len(missing_rows)}")
    print(missing_rows.head())
"""
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the cleaned dataset
df = pd.read_csv('Spotify_cleaned.csv', low_memory=False)

# 2. Select numeric columns to inspect for outliers
numeric_cols = [
    'Days',
    'Top 10 (xTimes)',
    'Peak Position',
    'Peak Position (xTimes)',
    'Peak Streams',
    'Total Streams'
]
# Plot boxplots to highlight potential outliers
for col in numeric_cols:
    plt.figure()
    df.boxplot(column=[col])
    plt.title(f'Boxplot of {col}')
    plt.ylabel(col)
    plt.show()
