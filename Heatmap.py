import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the cleaned Spotify data
df = pd.read_csv('Spotify_cleaned.csv')

# 2. Select the key numeric columns
numeric_cols = ['Days', 'Peak Streams', 'Peak Position', 'Total Streams']
numeric_df  = df[numeric_cols].copy()

# ---------------------------------------------------------
# A) Correlation heatmap
# ---------------------------------------------------------
corr = numeric_df.corr()

plt.figure(figsize=(6, 5))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar_kws={'label': 'Pearson r'}
)
plt.title("Correlation Matrix of Key Streaming Metrics")
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()


# ---------------------------------------------------------
# B) Distributions & Boxplots (Original + log-transformed)
# ---------------------------------------------------------
for col in numeric_cols:
    # ORIGINAL DISTRIBUTION & BOXPLOT
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    sns.histplot(numeric_df[col], bins=30, kde=True)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    sns.boxplot(x=numeric_df[col])
    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)

    plt.tight_layout()
    plt.savefig(f'{col}_hist_box.png', dpi=300, bbox_inches='tight')
    plt.show()

    # LOG-TRANSFORMED DISTRIBUTION & BOXPLOT
    log_col = f'log1p_{col}'
    numeric_df[log_col] = np.log1p(numeric_df[col])  # log(1 + x)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    sns.histplot(numeric_df[log_col], bins=30, kde=True)
    plt.title(f"Histogram of log1p({col})")
    plt.xlabel(log_col)
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    sns.boxplot(x=numeric_df[log_col])
    plt.title(f"Boxplot of log1p({col})")
    plt.xlabel(log_col)

    plt.tight_layout()
    plt.savefig(f'log1p_{col}_hist_box.png', dpi=300, bbox_inches='tight')
    plt.show()
