import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
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
log_reg = sm.Logit(y_train, X_train[subset_a]).fit()
print(log_reg.summary())

# predict
predictions = log_reg.predict(X_test[subset_a])
predictions = list(map(round,predictions))
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# SUBSET B
print("\n\nLOGISTIC REGRESSION SUBSET B")

# train
log_reg = sm.Logit(y_train, X_train[subset_b]).fit()
print(log_reg.summary())

# predict
predictions = log_reg.predict(X_test[subset_b])
predictions = list(map(round,predictions))
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# BAG OF WORDS
print("\n\nLOGISTIC REGRESSION BOW")

# pipeline
bow_pipe = make_pipeline(
    ColumnTransformer(remainder='passthrough',
                      transformers=[('countvectorizer',
                                     CountVectorizer(),
                                     'tokenized_words')]),
    StandardScaler(with_mean=False),
    LogisticRegression(penalty='l2',
                       solver='saga',
                       max_iter=200,
                       random_state=229))

# parameters to try
parameters = {
    'columntransformer__countvectorizer__max_features': (10000,50000),
    'logisticregression__C': (0.01, 0.001, 0.0001)
}

# perform validation
gs_bow_pipe = GridSearchCV(bow_pipe, 
                           parameters, 
                           cv=ShuffleSplit(n_splits=1, 
                                           test_size=0.15, 
                                           random_state=229), 
                           n_jobs=-1)
gs_bow_pipe.fit(X_train, y_train)
print(gs_bow_pipe.cv_results_)
print(gs_bow_pipe.best_params_)

# predict
predictions = log_reg.predict(X_test[subset_b])
predictions = list(map(round,predictions))
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# feature importance
coefficients = gs_bow_pipe.best_estimator_.named_steps['logisticregression'].coef_[0]
num_nonzero_coefs = len(np.where(abs(coefficients) > 0)[0])
sorted_ind = np.argsort(abs(coefficients))[::-1][:num_nonzero_coefs]
print(np.take(coefficients,sorted_ind.tolist()))
print(np.take(gs_bow_pipe.best_estimator_.named_steps['columntransformer'].get_feature_names(),sorted_ind.tolist()))