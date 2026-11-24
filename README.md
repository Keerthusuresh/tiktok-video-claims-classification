# TikTok Video Claims Classification

An end-to-end binary classification project that predicts whether a TikTok video contains a claim or an opinion. The aim is to support moderation-priority decisions, not to determine whether a claim is true, false, harmful, or policy-violating.

## Business objective

TikTok receives many user reports that may need moderator review. This project explores whether video metadata, engagement, account attributes, and transcription text can help prioritize videos likely to contain claims for human review.

## Dataset

The project uses the tiktok_dataset.csv dataset linked with the supplied Kaggle notebook:

- 19,382 video rows and 12 original columns
- Target: claim_status
- Usable labelled rows after dropping records without the target: 19,084
- Classes: claim and opinion

The dataset is synthetic. The raw dataset remains inside the data folder for reproducibility.

## Workflow

1. Inspect data quality, missing values, duplicate records, class balance, and potential outliers.
2. Perform EDA and visualizations of engagement, duration, author status, and class relationships.
3. Use a Welch two-sample t-test to compare verified and unverified view counts.
4. Engineer engagement-rate, text-length, and duration-band features.
5. Fit preprocessing only on training data through Scikit-learn pipelines.
6. Compare a baseline, Logistic Regression, Random Forest, and tuned Random Forest.
7. Evaluate using accuracy, precision, recall, F1-score, ROC-AUC, reports, matrices, curves, and train-versus-test performance.

## Actual test-set results

Positive class: claim.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.5035 | 0.5035 | 1.0000 | 0.6698 | N/A |
| Logistic Regression | 0.9995 | 1.0000 | 0.9990 | 0.9995 | 1.0000 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Tuned Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Random Forest is selected in this run. The tuning search produced a best three-fold training cross-validation F1 of approximately 0.9993.

## Key findings

- Median views: 501,555 for claim videos and 4,953 for opinion videos.
- Video view count is the most important fitted Random Forest feature.
- Strong performance needs cautious interpretation. Engagement may accumulate after publication and may not be available when a real moderation queue is created. This is a temporal-leakage risk.

## Run locally

Install dependencies from requirements.txt, then open TikTok_Claims_Classification_End_to_End.ipynb in Jupyter and run all cells from top to bottom. Keep the data folder alongside the notebook.

