import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the train/test predictions
results = pd.read_csv('regression_results.csv')

# 2. Prepare axis limits for the identity line
min_val = results[['actual', 'pred_lr', 'pred_rf']].min().min()
max_val = results[['actual', 'pred_lr', 'pred_rf']].max().max()

# 3. Define Spotify theme color
spotify_green = '#1DB954'

# 4. Create figure with Spotify-green background
plt.figure(figsize=(8, 6), facecolor=spotify_green)
ax = plt.gca()
ax.set_facecolor(spotify_green)

# 5. Scatter actual vs predicted for each model with white markers
ax.scatter(results['actual'], results['pred_lr'],
           label='Linear Regression',
           marker='o',
           color='white',
           alpha=0.7)
ax.scatter(results['actual'], results['pred_rf'],
           label='Random Forest',
           marker='^',
           color='white',
           alpha=0.7)

# 6. Plot identity line in white
ax.plot([min_val, max_val], [min_val, max_val],
        color='white', linestyle='--', linewidth=1)

# 7. Styling axis labels and title
ax.set_xlabel('Actual Total Streams', color='white', fontsize=12)
ax.set_ylabel('Predicted Total Streams', color='white', fontsize=12)
ax.set_title('Regression: Actual vs Predicted Streams', color='white', fontsize=14)

# 8. Tick and legend styling
ax.tick_params(colors='white')
legend = ax.legend()
for text in legend.get_texts():
    text.set_color('white')
legend.get_frame().set_facecolor(spotify_green)
legend.get_frame().set_edgecolor('white')

plt.tight_layout()
# 9. Show plot
plt.savefig('R_Model_Visual.png', facecolor=spotify_green, bbox_inches='tight')
plt.show()
