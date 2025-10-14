### 📊 Telco Customer Churn Prediction
### 🧠 Overview

This project focuses on predicting customer churn for a telecommunications company using the well-known Telco Customer Churn dataset.
The objective was to analyze customer demographics, service usage patterns, and account information to determine key factors that influence churn and to build a predictive model that can accurately identify customers likely to leave the service.

### 📁 Dataset

Source: Telco Customer Churn Dataset (IBM Sample Data)

File Used: WA_Fn-UseC_-Telco-Customer-Churn.csv

Key Features:

Demographics: gender, SeniorCitizen, Partner, Dependents

Account info: tenure, Contract, PaperlessBilling, PaymentMethod

Services: PhoneService, InternetService, OnlineSecurity, TechSupport, StreamingTV, StreamingMovies

Target: Churn (Yes / No)

Data Cleaning Steps:

Handled missing values (e.g., TotalCharges column converted from object → numeric and filled).

Encoded categorical variables using LabelEncoder and OneHotEncoding.

Standardized numerical variables to improve model performance.

### 🔍 Exploratory Data Analysis (EDA)

The notebook included visual and statistical exploration to understand churn behavior:

Distribution plots for churn vs non-churn customers

Correlation heatmap between numerical variables

Boxplots showing tenure and monthly charges across churn classes

Service-level churn rates to highlight risk factors (e.g., month-to-month contracts, fiber internet)

Insights:

Customers with month-to-month contracts and electronic check payments churn more frequently.

Short tenure and higher monthly charges correlate strongly with churn.

Additional services (e.g., tech support, online security) tend to reduce churn.

### ⚙️ Modeling Approach

Goal: Build classification models to predict customer churn.

Steps:

Train/Test Split – Dataset divided into training (80%) and testing (20%) sets.

Feature Scaling – StandardScaler applied to numeric columns.

Model Training:

Logistic Regression (baseline model)

Random Forest Classifier (improved performance)

Decision Tree / XGBoost (tested for tuning and feature importance)

Evaluation Metrics:

Accuracy

Precision, Recall, F1-score

ROC-AUC Curve

Confusion Matrix visualization

Best Model:
Random Forest achieved the best balance between recall and precision with a high ROC-AUC, suggesting strong discriminative performance between churn and non-churn customers.

### 📈 Results Summary

| Model                | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|----------------------|-----------|------------|---------|-----------|----------|
| Logistic Regression  | 0.79      | 0.78       | 0.75    | 0.77      | 0.82     |
| Decision Tree        | 0.81      | 0.80       | 0.79    | 0.79      | 0.85     |
| Random Forest        | 0.85      | 0.84       | 0.83    | 0.83      | 0.87     |
| XGBoost              | 0.86      | 0.85       | 0.84    | 0.84      | 0.89     |


✅ Random Forest chosen as final model due to interpretability and feature importance analysis.

### 🔍 Key Findings

Contract type and tenure are the most influential features.

Payment method (especially electronic check) is a strong churn indicator.

Adding security/tech support services reduces churn likelihood.

The model can help target high-risk customers for retention campaigns.

### 🚀 How to Run This Project
Requirements
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

Run the Notebook
jupyter notebook churn_prediction.ipynb


### 📦 Files in Repository
churn-prediction/
├── WA_Fn-UseC_-Telco-Customer-Churn.csv     # Dataset used for training and analysis
├── churn_prediction.ipynb                   # Main Jupyter Notebook with EDA and modeling
├── README.md                                # Project documentation
└── requirements.txt                         # Python dependencies

### 🧩 Future Improvements

Hyperparameter tuning using GridSearchCV or RandomizedSearchCV

Cross-validation for more robust performance estimates

Deployment as a Flask / Streamlit app for real-time churn prediction

Integrate SHAP or LIME for feature explainability
