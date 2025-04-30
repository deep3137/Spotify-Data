import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Load data
df = pd.read_csv('Spotify_cleaned.csv')

# 2. Create binary target: hit_top10 = 1 if Peak Position ≤ 10
df['hit_top10'] = (df['Peak Position'] <= 10).astype(int)

# 3. Feature matrix & label
X = df[['Days', 'Peak Streams', 'Total Streams']]
y = df['hit_top10']

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 5. Logistic Regression
clf_lr = LogisticRegression(max_iter=1000)
clf_lr.fit(X_train, y_train)
y_pred_lr = clf_lr.predict(X_test)
y_proba_lr = clf_lr.predict_proba(X_test)[:, 1]

# 6. Random Forest Classifier
clf_rf = RandomForestClassifier(random_state=42)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)
y_proba_rf = clf_rf.predict_proba(X_test)[:, 1]

# 7. Print performance
print("=== Logistic Regression ===")
print(classification_report(y_test, y_pred_lr))
print("ROC AUC:", roc_auc_score(y_test, y_proba_lr), "\n")

print("=== Random Forest ===")
print(classification_report(y_test, y_pred_rf))
print("ROC AUC:", roc_auc_score(y_test, y_proba_rf), "\n")

# 8. Save actuals, predictions, and probabilities
results = pd.DataFrame({
    'actual':   y_test.values,
    'pred_lr':  y_pred_lr,
    'proba_lr': y_proba_lr,
    'pred_rf':  y_pred_rf,
    'proba_rf': y_proba_rf
})
results.to_csv('classification_results.csv', index=False)
print("Saved classification_results.csv")
