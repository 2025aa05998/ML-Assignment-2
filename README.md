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
| Model               |   Accuracy |      AUC |   Precision |   Recall |       F1 |      MCC |
|:--------------------|-----------:|---------:|------------:|---------:|---------:|---------:|
| Logistic Regression |   0.95614  | 0.997707 |    0.975    | 0.906977 | 0.939759 | 0.906811 |
| Decision Tree       |   0.947368 | 0.94399  |    0.930233 | 0.930233 | 0.930233 | 0.887979 |
| KNN                 |   0.95614  | 0.995906 |    1        | 0.883721 | 0.938272 | 0.908615 |
| Naive Bayes         |   0.973684 | 0.998362 |    1        | 0.930233 | 0.963855 | 0.944733 |
| Random Forest       |   0.964912 | 0.995414 |    0.97561  | 0.930233 | 0.952381 | 0.925285 |
| XGBoost             |   0.95614  | 0.990829 |    0.952381 | 0.930233 | 0.941176 | 0.906379 |

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
