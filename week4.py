import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report

# 1. Generate Synthetic Dataset
np.random.seed(42)
n_samples = 1000

X1 = np.random.normal(loc=50, scale=15, size=n_samples) # Feature 1: Age
X2 = np.random.normal(loc=5000, scale=1500, size=n_samples) # Feature 2: Monthly Spend
X3 = np.random.exponential(scale=12, size=n_samples) # Feature 3: Tenure (months)

# Binary target (Churn: 0 or 1)
y = (0.03 * X1 - 0.0005 * X2 - 0.08 * X3 + np.random.normal(0, 1, n_samples) > 0).astype(int)

df = pd.DataFrame({'Age': X1, 'MonthlySpend': X2, 'Tenure': X3, 'Churn': y})

# 2. Train / Test Split & Preprocessing
X = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Training
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred = rf_model.predict(X_test_scaled)
y_probs = rf_model.predict_proba(X_test_scaled)[:, 1]

# Print Report
print(classification_report(y_test, y_pred))

# 4. Visualization 1: Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Stayed', 'Churned'])
disp.plot(cmap='Blues', ax=ax)
plt.title('Confusion Matrix - Random Forest Model')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

# 5. Visualization 2: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300)
plt.show()
