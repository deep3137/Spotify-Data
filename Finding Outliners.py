import pandas as pd
import matplotlib.pyplot as plt


def plot_combined_iqr_boxplots(filepath: str = 'Spotify_cleaned.csv'):
    """
    Load the cleaned Spotify data and generate all IQR-based boxplots
    in a single figure to highlight outliers.
    """
    # 1. Load data
    df = pd.read_csv(filepath, low_memory=False)

    # 2. Define numeric columns
    numeric_cols = [
        'Days',
        'Top 10 (xTimes)',
        'Peak Position',
        'Peak Position (xTimes)',
        'Peak Streams',
        'Total Streams'
    ]

    # 3. Compute and print outlier counts
    outlier_info = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        count = df[(df[col] < lower) | (df[col] > upper)].shape[0]
        outlier_info[col] = (count, lower, upper)
        print(f"{col}: {count} outliers (bounds: [{lower:.2f}, {upper:.2f}])")

    # 4. Plot all boxplots in one figure (2 rows x 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for ax, col in zip(axes, numeric_cols):
        ax.boxplot(
            df[col].dropna(),
            vert=True,
            whis=1.5,
            flierprops=dict(
                marker='o', markerfacecolor='red', markersize=4, linestyle='none'
            )
        )
        ax.set_title(f'{col}')
        ax.set_ylabel(col)

    plt.suptitle('IQR-based Boxplots for Numeric Features', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
    plt.show()


if __name__ == '__main__':
    # Ensure 'Spotify_cleaned.csv' is in the same directory as this script
    plot_combined_iqr_boxplots()
