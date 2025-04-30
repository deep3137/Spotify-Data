import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

# 1. Load the saved results
results = pd.read_csv('classification_results.csv')
y_true    = results['actual']
y_proba_lr = results['proba_lr']
y_proba_rf = results['proba_rf']
y_pred_rf = results['pred_rf']

# 2. Compute ROC curves
fpr_lr, tpr_lr, _ = roc_curve(y_true, y_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_true, y_proba_rf)

# 3. Spotify theme
spotify_green = '#1DB954'
plt.rcParams['axes.facecolor'] = spotify_green
plt.rcParams['figure.facecolor'] = spotify_green

# --- ROC Curve ---
plt.figure(figsize=(6,5))
ax = plt.gca()

# Plot curves
ax.plot(fpr_lr, tpr_lr,
        label=f'LogReg (AUC={auc(fpr_lr,tpr_lr):.2f})',
        color='white')
ax.plot(fpr_rf, tpr_rf,
        label=f'RF (AUC={auc(fpr_rf,tpr_rf):.2f})',
        color='white', linestyle='--')
# Identity line
ax.plot([0,1], [0,1],
        color='white', linestyle=':')

# Styling
ax.set_title('ROC Curves: Hit Top-10 Prediction', color='white')
ax.set_xlabel('False Positive Rate', color='white')
ax.set_ylabel('True Positive Rate', color='white')
ax.tick_params(colors='white')
leg = ax.legend()
for text in leg.get_texts():
    text.set_color('white')
leg.get_frame().set_facecolor(spotify_green)
leg.get_frame().set_edgecolor('white')

plt.tight_layout()
plt.savefig('classification_roc.png', dpi=300, facecolor=spotify_green)
plt.show()


# --- Confusion Matrix (Random Forest) ---
cm = confusion_matrix(y_true, y_pred_rf)
plt.figure(figsize=(4,3))
ax = plt.gca()

# Use a white edge for boxes on green
sns.heatmap(cm, annot=True, fmt='d',
            cmap='Greens', cbar=False,
            linecolor='white', linewidths=0.5,
            ax=ax)

# Styling
ax.set_title('RF Confusion Matrix', color='white')
ax.set_xlabel('Predicted', color='white')
ax.set_ylabel('Actual', color='white')
ax.tick_params(colors='white')
# Annotate numbers in white
for text in ax.texts:
    text.set_color('white')

plt.tight_layout()
plt.savefig('classification_cm.png', dpi=300, facecolor=spotify_green)
plt.show()
