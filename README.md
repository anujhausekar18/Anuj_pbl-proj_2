Breast Cancer Prediction using AI (Logistic Regression)
Overview

This project uses Machine Learning (Logistic Regression) to predict the survival status (Alive or Dead) of breast cancer patients based on various clinical and biological parameters.
The model is trained using real-world breast cancer data and generates key performance metrics to evaluate accuracy and identify the most influential features.

Key Features

Automated data preprocessing, encoding, and scaling

Logistic Regression-based prediction model

Detailed performance metrics including accuracy, confusion matrix, and classification report

Feature importance ranking for interpretability

Visualization of results through plots and tables

Tech Stack
Component	Library Used
Data Handling	pandas, numpy
Data Visualization	matplotlib, seaborn
Model Training	scikit-learn
Output Display	IPython.display (Markdown formatting)
Dataset

File Name: Breast Cancer Dataset(in).csv

Path: Update the file path before running the script.

file_path = r"C:\Users\Farahan\Downloads\Breast Cancer Dataset(in).csv"


The dataset should contain:

Patient characteristics (age, tumor size, lymph node count, etc.)

Clinical or diagnostic parameters

Target column: Status (with values such as Alive or Dead)

How It Works

Load Dataset – Reads the CSV file using pandas.

Data Encoding – Converts categorical variables to numeric using pd.get_dummies().

Feature-Target Split –

X = df_encoded.drop("Status_Dead", axis=1)
y = df_encoded["Status_Dead"]


Train-Test Split – Divides data into 80% training and 20% testing sets.

Feature Scaling – Standardizes input features using StandardScaler().

Model Training – Fits the logistic regression model on the scaled training data.

Prediction & Evaluation –
Generates:

Accuracy score

Confusion matrix

Classification report

Visualization – Displays heatmap and feature importance bar chart for interpretation.

Model Results

After training, the model outputs:

Accuracy: Displays the percentage of correctly predicted outcomes.

Confusion Matrix: Shows correct and incorrect predictions.

Classification Report: Includes precision, recall, and F1-score for each class.

Top Features: Identifies the top 10 most influential factors affecting survival predictions.

Example Output
Model Accuracy: 95.4 %

Confusion Matrix:
              Predicted: Alive   Predicted: Dead
Actual: Alive        72               3
Actual: Dead          2              43


Top 10 Important Features

Tumor_Size             0.892
Age                    0.744
Lymph_Nodes_Positive   0.615
...


Prediction Example:

Patient #1 → Predicted Status: Alive

Visualization Summary

Confusion Matrix Heatmap – Visual representation of model performance.

Feature Importance Chart – Displays which clinical features have the highest impact on prediction.

Learnings and Insights

Logistic Regression performs effectively for binary medical prediction problems.

Proper feature scaling improves model accuracy and stability.

Feature importance analysis provides meaningful clinical insights.

The workflow can easily be adapted to other healthcare prediction models.

Requirements

Make sure the following Python libraries are installed before running the code:

pip install pandas numpy scikit-learn matplotlib seaborn ipython

Future Improvements

Implement advanced models such as Random Forest, SVM, or XGBoost.

Apply hyperparameter tuning (GridSearchCV or RandomizedSearchCV).

Use explainability tools such as SHAP or LIME for deeper insight.

Deploy the model using Streamlit or Flask for interactive prediction.
