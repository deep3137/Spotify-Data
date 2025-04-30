import pandas as pd
import matplotlib.pyplot as plt


def load_cleaned_data(filename: str = 'Spotify_cleaned.csv') -> pd.DataFrame:
    """
    Load the already-cleaned Spotify dataset from the working directory.
    """
    return pd.read_csv(filename, low_memory=False)


def handle_outliers_iqr(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Clip each column in `cols` to the [Q1 - 1.5·IQR, Q3 + 1.5·IQR] bounds.
    Returns a new DataFrame with outliers capped.
    """
    df = df.copy()
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)
    return df


if __name__ == '__main__':
    # 1. Load cleaned data
    df = load_cleaned_data('Spotify_cleaned.csv')

    # 2. Specify numeric columns
    numeric_cols = [
        'Days',
        'Top 10 (xTimes)',
        'Peak Position',
        'Peak Position (xTimes)',
        'Peak Streams',
        'Total Streams'
    ]

    # 3. Cap outliers
    df_capped = handle_outliers_iqr(df, numeric_cols)

    # 4. Plot all boxplots in one figure, boxes colored blue
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax, col in zip(axes, numeric_cols):
        bp = ax.boxplot(
            df_capped[col].dropna(),
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor='blue', alpha=0.5),
            medianprops=dict(color='black'),
            flierprops=dict(marker='o', markerfacecolor='red', markersize=4, linestyle='none')
        )
        ax.set_title(col)
        ax.set_ylabel(col)

    plt.suptitle('IQR-Capped Boxplots for Spotify Metrics', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
