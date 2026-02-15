# Breast Cancer Classification (UCI Dataset)

## Problem Statement
The goal of this project is to classify tumors as malignant (M) or benign (B) using multiple machine learning models. The assignment demonstrates end-to-end ML workflow: dataset preparation, model training, evaluation, and deployment via a Streamlit app.

## Dataset Description
- Source: UCI ML Repository – Breast Cancer Wisconsin (Diagnostic)
- Instances: 569
- Features: 30 real-valued features computed from digitized images of fine needle aspirates of breast masses
- Target: Malignant (M=1) vs Benign (B=0)

## Models Used
- Logistic Regression
- Decision Tree
- KNN
- Naive Bayes
- Random Forest
- XGBoost

## Comparison Table (Evaluation Metrics)
{results_df.to_markdown(index=False)}

## Observations on Model Performance
- Logistic Regression: Strong baseline with high accuracy and excellent AUC.
- Decision Tree: Good accuracy but slightly lower AUC, shows signs of overfitting.
- KNN: Performance drops significantly, especially recall.
- Naive Bayes: Very poor precision and recall, independence assumptions don’t hold.
- Random Forest: Excellent performance, robust and generalizes well.
- XGBoost: Best overall metrics with highest AUC, balances precision and recall effectively.

## Streamlit App Features
- Dataset upload option (CSV)
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix or classification report

## Repository Structure
project-folder/
├── app.py
├── requirements.txt
├── README.md
└── model/
    ├── Logistic_Regression.pkl
    ├── Decision_Tree.pkl
    ├── KNN.pkl
    ├── Naive_Bayes.pkl
    ├── Random_Forest.pkl
    └── XGBoost.pkl
