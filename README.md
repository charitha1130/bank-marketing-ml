# Bank Marketing - Term Deposit Prediction

- Live app: https://bits-ml-assignment-2-bank-marketing.streamlit.app/
- GitHub: https://github.com/charitha1130/bank-marketing-ml

## a. Problem statement

A Portuguese bank sells term deposits through phone campaigns. Calling every
customer costs time and money, so it helps to know in advance which clients are
likely to say yes. Given a client's profile and the history of the campaign, the
task is to predict whether the client subscribes to a term deposit (`yes`) or
not (`no`).

This is a binary classification problem. Five classifiers (Logistic Regression,
Decision Tree, KNN, Naive Bayes and SVM) are trained on the same data, compared
using six metrics, and shown in a Streamlit app where a test csv can be uploaded.

## b. Dataset description

The data is the Bank Marketing dataset from the UCI Machine Learning Repository
(`bank-full.csv`). Citation: Moro, S., Rita, P., & Cortez, P. (2014). Bank
Marketing. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306

| Item | Value |
|------|-------|
| Instances | 45,211 |
| Input features | 16 (7 numeric, 9 categorical) |
| Target | `y` - client subscribed to a term deposit (yes / no) |
| Class balance | about 88% no, 12% yes |
| Missing values | none (unknown values are stored as the text "unknown") |

| Feature | Type | Meaning |
|---------|------|---------|
| age | numeric | age of the client |
| job | categorical | type of job |
| marital | categorical | marital status |
| education | categorical | education level |
| default | categorical | has credit in default |
| balance | numeric | average yearly balance in euros |
| housing | categorical | has a housing loan |
| loan | categorical | has a personal loan |
| contact | categorical | contact type (cellular, telephone, unknown) |
| day | numeric | day of the month of the last contact |
| month | categorical | month of the last contact |
| duration | numeric | length of the last call in seconds |
| campaign | numeric | number of contacts in this campaign |
| pdays | numeric | days since the client was contacted in an earlier campaign (-1 = never) |
| previous | numeric | number of contacts before this campaign |
| poutcome | categorical | outcome of the earlier campaign |

**Preprocessing.** The data is split 80/20 with a stratified split
(`random_state=42`), giving 36,168 training rows and 9,043 test rows. Numeric
columns are scaled with `StandardScaler` and categorical columns are one-hot
encoded. The preprocessor is fitted on the training part only and saved as
`model/preprocessor.pkl`. All five models use the same preprocessed features.

## c. GitHub repository link

https://github.com/charitha1130/bank-marketing-ml

Live Streamlit app: https://bits-ml-assignment-2-bank-marketing.streamlit.app/

## d. Models used

Results on the 20% hold-out test set (9,043 rows, saved as `test_data.csv`):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.9012 | 0.9056 | 0.6445 | 0.3478 | 0.4518 | 0.4261 |
| Decision Tree | 0.9007 | 0.8771 | 0.6201 | 0.3904 | 0.4791 | 0.4414 |
| KNN | 0.8986 | 0.8500 | 0.6257 | 0.3318 | 0.4336 | 0.4070 |
| Naive Bayes | 0.8548 | 0.8101 | 0.4059 | 0.5198 | 0.4559 | 0.3774 |
| Support Vector Machine | 0.9049 | 0.9036 | 0.6875 | 0.3431 | 0.4578 | 0.4420 |

Precision, recall and F1 are for the positive class (`yes`). AUC is computed
from predicted probabilities; the SVM has no probabilities here, so its
decision function score is used. Because about 88% of clients said no, a model
that always predicts "no" already reaches about 0.88 accuracy. Accuracy alone
is therefore not very informative, and AUC, recall, F1 and MCC matter more.

Settings used for each model:

| Model | Settings |
|-------|----------|
| Logistic Regression | max_iter=1000 |
| Decision Tree | max_depth=8, min_samples_leaf=20 |
| KNN | n_neighbors=7 |
| Naive Bayes | GaussianNB, default settings |
| SVM | RBF kernel, C=1.0 |

### Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Best AUC (0.9056), so it ranks likely subscribers well. Recall is low (0.35): at the default 0.5 cut-off it misses most of the real "yes" clients. |
| Decision Tree | Best F1 (0.4791) and better recall (0.39) than Logistic Regression and SVM. AUC (0.8771) is lower, so its probability ranking is rougher. Depth was limited to 8 to control overfitting. |
| KNN | Weakest of the balanced models: lowest recall (0.33), lowest F1 (0.4336) and AUC of only 0.85. Distances get diluted by the many one-hot columns. |
| Naive Bayes | Lowest accuracy (0.8548), precision (0.4059), AUC and MCC, but the highest recall (0.5198). It labels many more clients as "yes", which catches more buyers but gives many false positives. The independence assumption does not hold well for one-hot features. |
| Support Vector Machine | Highest accuracy (0.9049), precision (0.6875) and MCC (0.4420), with AUC (0.9036) almost equal to Logistic Regression. Recall is low (0.34), like Logistic Regression. |
| **Overall, winner for your dataset?** | SVM is the best overall on accuracy, precision and MCC, with Logistic Regression close behind on AUC. If the bank cares more about catching subscribers, Decision Tree (F1) or Naive Bayes (recall) fit better. Recall is low for most models because of the class imbalance. |

## Streamlit app

The app loads the saved models from `model/` and does not retrain on the data
that is uploaded.

- Upload a test csv from the sidebar (use `test_data.csv`)
- Dropdown to select one of the five models
- Accuracy, AUC, Precision, Recall, F1 and MCC for the selected model
- Confusion matrix and classification report
- A comparison table and chart with all five models on the same file
- A preview of the uploaded data

The uploaded file needs the 16 input columns and the target column `y`
(0/1 or no/yes). If nothing is uploaded, the app shows the bundled
`test_data.csv`.

## Repository contents

```
bank-marketing-ml/
|-- app.py                   Streamlit app
|-- train_models.py          training and evaluation script
|-- bank_marketing_ml.ipynb  notebook version of the same steps
|-- requirements.txt         pinned packages
|-- test_data.csv            hold-out test rows with the target column
|-- model_results.csv        metrics table from training
|-- data/                    original UCI files
|-- model/                   saved preprocessor and the five trained models
|-- .streamlit/config.toml   app theme
```

## How to run

```
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

`train_models.py` recreates `model/`, `test_data.csv` and `model_results.csv`.

## About the csv files

- `data/bank-full.csv` is the full UCI file. 80% of it was used for training.
- `test_data.csv` is the other 20% (9,043 rows) with `y` stored as 0/1. This is
  the file to upload in the app, since the models never saw it in training.
- The full `bank-full.csv` also works in the app, but its scores look better
  than they should because it contains the training rows.

## Notes on the data

- The classes are imbalanced (about 12% `yes`). No resampling or threshold
  tuning was done, which is the main reason recall is low for most models.
- `duration` (length of the last call) is only known after the call is made, so
  it could not be used in a real campaign. It was kept to stay close to the
  standard version of the dataset.
