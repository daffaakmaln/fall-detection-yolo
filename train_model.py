import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("dataset_fitur_balanced.csv")
print(f"Total data: {len(df)}")
print(f"Fall: {len(df[df['label']=='fall'])}, Normal: {len(df[df['label']=='normal'])}")

# fitur = x, label = y
X = df[["sudut", "kecepatan"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# train with randum furest
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# cross validation
print("\n── CROSS VALIDATION (5-Fold) ──")
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Akurasi per fold: {[f'{s:.2%}' for s in cv_scores]}")
print(f"Rata-rata       : {cv_scores.mean():.2%}")
print(f"Standar deviasi : {cv_scores.std():.2%}")

# test data evaluationn
y_pred = model.predict(X_test)

print("\n── CLASSIFICATION REPORT ──")
print(classification_report(y_test, y_pred))

print("\n── FEATURE IMPORTANCE ──")
for fitur, importance in zip(X.columns, model.feature_importances_):
    print(f"  {fitur}: {importance:.2%}")

# export confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["fall", "normal"])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["fall", "normal"], yticklabels=["fall", "normal"])
plt.title("Confusion Matrix - Random Forest")
plt.ylabel("Label Sebenarnya")
plt.xlabel("Label Prediksi")
plt.tight_layout()
plt.savefig("confusion_matrix_rf.png", dpi=150)
plt.show()

# sv model
joblib.dump(model, "model_fall_detection.pkl")
print("\n✅ Model tersimpan sebagai model_fall_detection.pkl")