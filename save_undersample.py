import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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

# undersample train set
majority_size = len(y_train[y_train==0])
minority_size = len(y_train[y_train==1])
majority_indices = y_train[y_train==0].index
rng = np.random.default_rng(seed=229)
drop_indices = rng.choice(majority_indices, majority_size-minority_size, replace=False)
X_train = X_train.drop(drop_indices)
y_train = y_train.drop(drop_indices)


X_train.to_pickle("data/X_train.pkl",protocol=3)
X_test.to_pickle("data/X_test.pkl",protocol=3)
y_train.to_pickle("data/y_train.pkl",protocol=3)
y_test.to_pickle("data/y_test.pkl",protocol=3)
