import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

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
print("RANDOM FOREST SUBSET A")

# model
rf = xgb.XGBRegressor(objective='binary:logistic',
                      eval_metric='error',
                      seed=229,
                      n_jobs=-1)

# parameters to try
parameters = {
    'n_estimators': (50,100,1000),
    'max_depth': (2,4,6),
    'learning_rate': (0.01, 0.1, 0.3)
}

# perform validation
gs_rf = GridSearchCV(rf,
                     parameters,
                     cv=ShuffleSplit(n_splits=1,
                                     test_size=0.13,
                                     random_state=229))
gs_rf.fit(X_train[subset_a], y_train)
print(gs_rf.cv_results_)
print(gs_rf.best_params_)

# predict
predictions = gs_rf.predict(X_test[subset_a])
predictions = list(map(round,predictions))
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(roc_auc_score(y_test, predictions))

# feature importance
print(gs_rf.best_estimator_.feature_importances_)
xgb.plot_importance(gs_rf.best_estimator_)
plt.tight_layout()
plt.savefig("rf_full_a.png")

# SUBSET B
print("\n\nRANDOM FOREST SUBSET B")

# model, parameters to try, gridsearch defined above

# perform validation
gs_rf.fit(X_train[subset_b], y_train)
print(gs_rf.cv_results_)
print(gs_rf.best_params_)

# predict
predictions = gs_rf.predict(X_test[subset_b])
predictions = list(map(round,predictions))
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(roc_auc_score(y_test, predictions))

# feature importance
print(gs_rf.best_estimator_.feature_importances_)
xgb.plot_importance(gs_rf.best_estimator_)
plt.tight_layout()
plt.savefig("rf_full_b.png")