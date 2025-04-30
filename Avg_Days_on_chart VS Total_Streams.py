import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the clustered data
artist_stats = pd.read_csv('artist_hit_profiles_with_clusters.csv')

# 2. Spotify theme color
spotify_green = '#1DB954'

# 3. Create figure & axes with green background
plt.figure(figsize=(10, 6), facecolor=spotify_green)
ax = plt.gca()
ax.set_facecolor(spotify_green)

# 4. Scatter by cluster
for cluster_label in sorted(artist_stats['cluster'].unique()):
    subset = artist_stats[artist_stats['cluster'] == cluster_label]
    ax.scatter(
        subset['avg_days_on_chart'],
        subset['total_streams'],
        label=f'Cluster {cluster_label}',
        alpha=0.7,
        edgecolors='white'
    )

# 5. Styling
ax.set_xlabel('Average Days on Chart', color='white', fontsize=12)
ax.set_ylabel('Total Streams', color='white', fontsize=12)
ax.set_title('Artist Hit-Profile Clusters', color='white', fontsize=14)
ax.tick_params(colors='white')

legend = ax.legend()
for text in legend.get_texts():
    text.set_color('white')
legend.get_frame().set_facecolor(spotify_green)

plt.tight_layout()

# 6. Save and show
plt.savefig('spotify_artist_clusters.png', dpi=300, facecolor=spotify_green)
plt.show()
