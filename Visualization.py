import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned Spotify data
df = pd.read_csv('Spotify_cleaned.csv')

# 1. Column chart: Top 10 Songs by Total Streams
song_streams = (
    df.groupby('Song Name')['Total Streams']
      .sum()
      .sort_values(ascending=False)
      .head(10)
)
plt.figure(figsize=(10, 6))
plt.bar(song_streams.index, song_streams.values)
plt.xlabel('Song Name')
plt.ylabel('Total Streams')
plt.title('Top 10 Songs by Total Streams')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('column_chart.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Bar chart: Top 10 Artists by #1 Peaks
artist_peaks = (
    df[df['Peak Position'] == 1]['Artist Name']
      .value_counts()
      .head(10)
)
plt.figure(figsize=(8, 6))
plt.barh(artist_peaks.index, artist_peaks.values)
plt.xlabel('Number of #1 Peaks')
plt.ylabel('Artist Name')
plt.title('Top 10 Artists by #1 Peak Positions')
plt.tight_layout()
plt.savefig('bar_chart.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Pie chart: Proportion of Songs Hitting Top-10
hits = df['Peak Position'].le(10).value_counts()
labels = ['Hit Top 10', 'Did Not Hit Top 10']
plt.figure(figsize=(6, 6))
plt.pie(hits.values, labels=labels, autopct='%1.1f%%')
plt.title('Songs Hitting Top-10 vs Not')
plt.savefig('pie_chart.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Box-plot: Distribution of Total Streams
plt.figure(figsize=(8, 4))
plt.boxplot(df['Total Streams'], vert=False)
plt.xlabel('Total Streams')
plt.title('Boxplot of Total Streams')
plt.tight_layout()
plt.savefig('boxplot_total_streams.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Line plot: Average Total Streams by Days on Chart
avg_streams = df.groupby('Days')['Total Streams'].mean()
plt.figure(figsize=(8, 5))
plt.plot(avg_streams.index, avg_streams.values)
plt.xlabel('Days on Chart')
plt.ylabel('Average Total Streams')
plt.title('Average Total Streams by Days on Chart')
plt.tight_layout()
plt.savefig('line_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Multi-line plot: Avg Peak Streams vs Avg Total Streams by Days
avg_metrics = df.groupby('Days')[['Peak Streams', 'Total Streams']].mean()
plt.figure(figsize=(8, 5))
plt.plot(avg_metrics.index, avg_metrics['Peak Streams'], label='Avg Peak Streams')
plt.plot(avg_metrics.index, avg_metrics['Total Streams'], label='Avg Total Streams')
plt.xlabel('Days on Chart')
plt.ylabel('Average Streams')
plt.title('Avg Peak vs Total Streams by Days on Chart')
plt.legend()
plt.tight_layout()
plt.savefig('multi_line_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. Scatter plot: Peak Streams vs Total Streams
plt.figure(figsize=(8, 6))
plt.scatter(df['Peak Streams'], df['Total Streams'], alpha=0.5)
plt.xlabel('Peak Streams')
plt.ylabel('Total Streams')
plt.title('Scatter: Peak Streams vs Total Streams')
plt.tight_layout()
plt.savefig('scatter_plot.png', dpi=300, bbox_inches='tight')
plt.show()
