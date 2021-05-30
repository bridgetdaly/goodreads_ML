import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

np.set_printoptions(threshold=sys.maxsize)

# read/prep data
dat = pd.read_csv("data/tokenized_reviews.csv")
dat = dat.dropna()
dat["quote"] = dat["quote"].astype(int)
dat["tokenized_words"] = dat["tokenized_words"].apply(lambda x: x.strip("[']").replace("', '"," "))


# 85% train / 15% test
X_train, X_test, y_train, y_test = train_test_split(dat.drop(columns=["popular"]), 
                                                    dat["popular"],
                                                    test_size = 0.15,
                                                    random_state = 229)

# feature subsets
subset_a = ["user_reviews","days_since_review","user_rating","rating_diff"]
subset_b = ["user_reviews","days_since_review","user_rating","rating_diff",
            "num_words","avg_word_len","avg_sent_len","pct_verbs",
            "pct_nouns","pct_adj","quote","sentiment"]

# SUBSET A
print("LOGISTIC REGRESSION SUBSET A")

# train
log_reg = sm.Logit(y_train, sm.add_constant(X_train[subset_a])).fit()
print(log_reg.summary())

# predict
predictions = log_reg.predict(sm.add_constant(X_test[subset_a]))
predictions = list(map(round,predictions))
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(roc_auc_score(y_test, predictions))

# SUBSET B
print("\n\nLOGISTIC REGRESSION SUBSET B")

# train
log_reg = sm.Logit(y_train, sm.add_constant(X_train[subset_b])).fit()
print(log_reg.summary())

# predict
predictions = log_reg.predict(sm.add_constant(X_test[subset_b]))
predictions = list(map(round,predictions))
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(roc_auc_score(y_test, predictions))

# UNDERSAMPLE
print("\n\nUNDERSAMPLE")

# undersample train set
majority_size = len(y_train[y_train==0])
minority_size = len(y_train[y_train==1])
majority_indices = y_train[y_train==0].index
rng = np.random.default_rng(seed=229)
drop_indices = rng.choice(majority_indices, majority_size-minority_size, replace=False)
X_train = X_train.drop(drop_indices)
y_train = y_train.drop(drop_indices)

# SUBSET A
print("LOGISTIC REGRESSION SUBSET A")

# train
log_reg = sm.Logit(y_train, sm.add_constant(X_train[subset_a])).fit()
print(log_reg.summary())

# predict
predictions = log_reg.predict(sm.add_constant(X_test[subset_a]))
predictions = list(map(round,predictions))
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(roc_auc_score(y_test, predictions))

# SUBSET B
print("\n\nLOGISTIC REGRESSION SUBSET B")

# train
log_reg = sm.Logit(y_train, sm.add_constant(X_train[subset_b])).fit()
print(log_reg.summary())

# predict
predictions = log_reg.predict(sm.add_constant(X_test[subset_b]))
predictions = list(map(round,predictions))
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(roc_auc_score(y_test, predictions))
