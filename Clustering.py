import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Load your cleaned Spotify data
df = pd.read_csv('Spotify_cleaned.csv')

# 2. Aggregate metrics per artist
#    – Days on chart: average
#    – Peak Streams: average
#    – Total Streams: sum
artist_stats = (
    df
    .groupby('Artist Name')[['Days', 'Peak Streams', 'Total Streams']]
    .agg({'Days': 'mean', 'Peak Streams': 'mean', 'Total Streams': 'sum'})
    .rename(columns={
        'Days': 'avg_days_on_chart',
        'Peak Streams': 'avg_peak_streams',
        'Total Streams': 'total_streams'
    })
)

# 3. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(artist_stats)

# 4. Run k-means
k = 4  # choose the number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
artist_stats['cluster'] = kmeans.fit_predict(X_scaled)

# 5. Inspect cluster sizes
print("Cluster membership counts:")
print(artist_stats['cluster'].value_counts(), "\n")

# 6. Inverse-transform centroids back to original scale for interpretation
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroid_df = pd.DataFrame(
    centroids,
    columns=['avg_days_on_chart','avg_peak_streams','total_streams']
).assign(cluster=range(k))

print("Cluster centroids (in original units):")
print(centroid_df)



# 7. (Optional) Save the artist_stats with cluster labels for further analysis
artist_stats.to_csv('artist_hit_profiles_with_clusters.csv')
