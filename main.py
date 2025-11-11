# BREAST CANCER PREDICTION USING AI (Logistic Regression)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown  # For clean, formatted output

file_path = r"C:\Users\Anuj\Downloads\Breast Cancer Dataset(in).csv  # Update path if needed
df = pd.read_csv(file_path)

display(Markdown("**Data Loaded Successfully**"))
display(df.head())

df_encoded = pd.get_dummies(df, drop_first=True)

# Define features (X) and target (y)
X = df_encoded.drop("Status_Dead", axis=1, errors='ignore')
y = df_encoded["Status_Dead"]

X = X.fillna(X.median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T

display(Markdown(f"### Model Accuracy: {round(accuracy * 100, 2)} %"))
display(Markdown("### Confusion Matrix:"))
display(pd.DataFrame(cm, columns=['Predicted: Alive', 'Predicted: Dead'], 
                     index=['Actual: Alive', 'Actual: Dead']))

display(Markdown("### Classification Report:"))
display(report_df)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', cbar=False)
plt.title("Confusion Matrix: Breast Cancer Prediction")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

feature_importance = pd.Series(model.coef_[0], index=X.columns)
top_features = feature_importance.abs().sort_values(ascending=False).head(10)

display(Markdown("### Top 10 Most Important Features"))
display(top_features)

plt.figure(figsize=(8, 4))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title("Top 10 Important Features")
plt.xlabel("Coefficient Value (Impact Strength)")
plt.ylabel("Feature")
plt.show()

display(Markdown("### Example Prediction"))
example = X_test.iloc[0:1]
pred = model.predict(scaler.transform(example))
display(Markdown(f"**Prediction:** {'Dead' if pred[0] == 1 else 'Alive'}"))
display(example)
