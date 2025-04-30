import pandas as pd

# 1. Load the cleaned data
df = pd.read_csv('Spotify_cleaned.csv')

# 2. Filter to only songs that reached #1
peak1 = df[df['Peak Position'] == 1]

# 3. Count how many #1 peaks each artist has, then take the top 20
peak1_counts = peak1['Artist Name'].value_counts().head(20)

# 4. For each of those top artists, find their best-peaking song
results = []
for artist, count in peak1_counts.items():
    artist_songs = df[df['Artist Name'] == artist]
    # pick the song with the lowest numeric Peak Position
    best = artist_songs.loc[artist_songs['Peak Position'].idxmin()]
    results.append({
        'artist': artist,
        'num_#1_peaks': count,
        'best_peak_position': int(best['Peak Position']),
        'song_name': best['Song Name']
    })

# 5. Print the results
print("Top 20 Artists by #1 Peaks and Their Best-Peaking Song\n")
for r in results:
    print(f"{r['artist']}: {r['num_#1_peaks']} × #1 peaks, "
          f"Best Peak = {r['best_peak_position']} (\"{r['song_name']}\")")
