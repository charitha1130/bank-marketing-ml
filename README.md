# Bank Marketing - Term Deposit Classification

ML Assignment 2 - M.Tech (AIML), BITS Pilani WILP

## a. Problem statement

A Portuguese bank runs phone marketing campaigns to sell term deposits. Calling
every customer is expensive, so the goal is to predict from the client's
details and the campaign history whether a client will subscribe to a term
deposit (`yes`) or not (`no`). This is a binary classification problem. Five
classifiers are trained on the same data, compared on six metrics and shown in
a Streamlit app.

## b. Dataset description

- Source: UCI Machine Learning Repository - Bank Marketing (`bank-full.csv`)
- Instances: 45,211
- Features: 16 (7 numeric, 9 categorical) + target `y`
- Target: `y` - has the client subscribed to a term deposit (yes/no)
- Class balance: about 88% `no` and 12% `yes`, so the data is imbalanced
- Missing values: none (unknown values are stored as the text "unknown")
- Citation: Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing. UCI Machine
  Learning Repository. https://doi.org/10.24432/C5K306

| Type | Columns |
|------|---------|
| Numeric | age, balance, day, duration, campaign, pdays, previous |
| Categorical | job, marital, education, default, housing, loan, contact, month, poutcome |

Preprocessing: stratified 80/20 train-test split (`random_state=42`), numeric
columns scaled with StandardScaler, categorical columns one-hot encoded. The
preprocessor is fit on the training data only and saved as
`model/preprocessor.pkl`.

## c. GitHub repository link

<PASTE GITHUB REPO LINK HERE>

Live app: <PASTE STREAMLIT APP LINK HERE>

## d. Models used

Comparison on the 20% hold-out test set (9,043 rows, same as `test_data.csv`):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.9012 | 0.9056 | 0.6445 | 0.3478 | 0.4518 | 0.4261 |
| Decision Tree | 0.9007 | 0.8771 | 0.6201 | 0.3904 | 0.4791 | 0.4414 |
| KNN | 0.8986 | 0.8500 | 0.6257 | 0.3318 | 0.4336 | 0.4070 |
| Naive Bayes | 0.8548 | 0.8101 | 0.4059 | 0.5198 | 0.4559 | 0.3774 |
| Support Vector Machine | 0.9049 | 0.9036 | 0.6875 | 0.3431 | 0.4578 | 0.4420 |

Precision, recall and F1 are for the positive class (`yes`). AUC uses predicted
probabilities, and the SVM uses its decision function score. Since about 88% of
the clients said "no", a model that always predicts "no" already gets about 0.88
accuracy, so accuracy alone does not say much here. AUC, recall, F1 and MCC are
more useful for comparing the models.

### Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Best AUC (0.9056), so it ranks clients who will subscribe well. Recall is low (0.35), it misses most of the real "yes" clients at the default 0.5 cut-off. |
| Decision Tree | Best F1 (0.4791) and a better recall (0.39) than LR and SVM. Its AUC (0.8771) is lower, so the probability ranking is rougher. Depth was limited to 8 to avoid overfitting. |
| KNN | Weakest of the balanced models: lowest recall (0.33), lowest F1 (0.4336) and AUC of only 0.85. Distances get diluted by the many one-hot columns. |
| Naive Bayes | Lowest accuracy (0.8548), precision (0.4059), AUC and MCC, but highest recall (0.5198). It flags many more clients as "yes", which catches more buyers but also gives many false positives. The independence assumption does not hold for one-hot features. |
| Support Vector Machine | Highest accuracy (0.9049), precision (0.6875) and MCC (0.4420), and AUC (0.9036) is almost the same as LR. Recall is low (0.34) like LR. |
| **Best for this dataset** | SVM is the best overall on accuracy, precision and MCC, with LR close behind on AUC. If the bank cares more about catching subscribers, Decision Tree (F1) or Naive Bayes (recall) fit better. All models have low recall because of the class imbalance. |

### Notes on the data

- Classes are imbalanced (about 12% `yes`). No resampling or threshold tuning was
  done, which is why recall is low for most models.
- `duration` (length of the last call) is only known after the call is made, so
  in a real campaign it could not be used for the prediction. It was kept to stay
  close to the standard version of the dataset.

## Project structure

```
bank-marketing-ml/
|-- app.py               streamlit app
|-- train_models.py      training + evaluation script
|-- requirements.txt
|-- README.md
|-- test_data.csv        hold-out test rows (with target column)
|-- model_results.csv    metrics table produced by train_models.py
|-- data/                original UCI files
|-- model/               saved preprocessor and the 5 trained models
```

## How to run

```
pip install -r requirements.txt
python train_models.py        # trains and saves models in model/
streamlit run app.py
```

## Streamlit app features

- Upload a test csv (use `test_data.csv`)
- Dropdown to pick one of the 5 models
- Accuracy, AUC, Precision, Recall, F1 and MCC for the selected model
- Confusion matrix and classification report
- Table with all 5 models on the uploaded data

The app loads the saved models and does not retrain.

### Which csv is which?

- `data/bank-full.csv` is the full UCI file the models were trained on (80% of it).
- `test_data.csv` is the 20% hold-out part (9,043 rows) with the `y` column as
  0/1. Upload this one in the app. The models never saw it during training.
- Uploading the full `bank-full.csv` also works, but the score will be inflated
  because it contains the training rows.
