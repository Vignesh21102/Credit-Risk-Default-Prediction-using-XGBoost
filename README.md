Credit Risk Default Prediction using XGBoost
Project Overview

This project predicts loan default risk using the Home Credit Default Risk dataset.

The objective is to identify high-risk loan applicants and support credit risk decision-making.

Technologies
Python
Pandas
NumPy
Scikit-Learn
XGBoost
Matplotlib
Seaborn
🎯 Business Problem

Financial institutions face major losses when borrowers fail to repay loans.

The goal of this project is to:

Predict the probability of loan default.
Identify high-risk customers before loan approval.
Support data-driven lending decisions.
Reduce financial losses through proactive risk assessment.

Dataset
Home Credit Default Risk Dataset
Binary Target:
0 = No Default
1 = Default

Missing Value Analysis
Identified columns with excessive missing values.
Removed columns containing more than 60% missing data.
Applied imputation techniques:
Median imputation for numerical features.
Most frequent imputation for categorical features.
Class Imbalance

Loan defaults represent a small portion of total applications.
Feature Engineering
Applicant Age
Employment Years
Credit-to-Income Ratio
Annuity-to-Income Ratio
Model
XGBoost Classifier
Class Imbalance Handling
One-Hot Encoding
Median/Mode Imputation
Evaluation Metric
ROC-AUC Score

Model Pipeline

The pipeline includes:

Missing Value Imputation
One-Hot Encoding
XGBoost Training


Show more lines
📌 Key Business Insights

The analysis helps identify:

High-risk borrowers.
Impact of income level on default probability.
Impact of education levels on repayment behavior.
Important financial indicators driving credit risk.

Potentially important predictors include:

Credit Amount
Income Level
Employment Duration
Age
Credit-to-Income Ratio
Annuity-to-Income Ratio






Results
Metric	ValueROC-AUC	0.75+
Model	XGBoost
Task	Credit Risk Classification
