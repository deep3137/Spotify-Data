import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the cleaned data
df = pd.read_csv('Spotify_cleaned.csv')

# 2. Filter to only #1 peaks and count per artist
peak1 = df[df['Peak Position'] == 1]
peak1_counts = peak1['Artist Name'].value_counts().head(20)

# 3. Find each top artist’s best-peaking song
best_songs = (
    df
    .loc[df['Artist Name'].isin(peak1_counts.index)]
    .groupby('Artist Name')
    .apply(lambda grp: grp.loc[grp['Peak Position'].idxmin(), 'Song Name'])
    .reindex(peak1_counts.index)
)

# 4. Plot as a bar chart on a Spotify-green background
spotify_green = '#1DB954'

plt.figure(figsize=(12, 8), facecolor=spotify_green)
ax = plt.gca()
ax.set_facecolor(spotify_green)

# draw the bars
bars = ax.bar(peak1_counts.index, peak1_counts.values, color='white')

# styling axes
ax.set_xticks(range(len(peak1_counts)))
ax.set_xticklabels(peak1_counts.index, rotation=45, ha='right', color='white', fontsize=10)
ax.tick_params(axis='y', colors='white')
ax.set_ylabel('Number of #1 Peaks', color='white', fontsize=12)
ax.set_title('Top 20 Artists by #1 Peaks\n(Annotated with Their Best-Peaking Song)',
             color='white', fontsize=14)

# annotate each bar with the song name
for bar, song in zip(bars, best_songs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, song,
            rotation=90, ha='center', va='bottom', fontsize=8, color='white')

plt.tight_layout()
plt.savefig('Top20_artists.png', facecolor=spotify_green, bbox_inches='tight')
plt.show()
