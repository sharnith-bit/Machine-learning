import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report,
confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("email.csv")
X = df.drop(columns=['Email No.', 'Prediction'])
y = df['Prediction']
X_sample = X.sample(n=3000, random_state=42)
y_sample = y.loc[X_sample.index]
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample,
test_size=0.2, random_state=42)
pipeline = Pipeline(steps=[
('svd', TruncatedSVD(n_components=100, random_state=42)),
('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'],
yticklabels=['Ham', 'Spam'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Email Classification")
plt.tight_layout()
plt.show()
