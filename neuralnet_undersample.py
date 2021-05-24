import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
import shap
import seaborn as sns
sns.set_theme(style="whitegrid")

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

# feature subsets
subset_a = ["user_reviews","days_since_review","user_rating","rating_diff"]
subset_b = ["user_reviews","days_since_review","user_rating","rating_diff",
            "num_words","avg_word_len","avg_sent_len","pct_verbs",
            "pct_nouns","pct_adj","quote","sentiment"]


# SUBSET A
print("NEURAL NET SUBSET A")

# build neural net
mod = Sequential()
# input layer
mod.add(Dense(units=5, input_dim=X_train[subset_a].shape[1], activation='relu'))
# first hidden layer
mod.add(Dense(units=4, activation='relu'))
# output layer
mod.add(Dense(units=1, activation='sigmoid'))

mod.compile(loss='binary_crossentropy'
            , optimizer='adam'
            , metrics=['accuracy'])

# https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

# fit neural net
history = mod.fit(x=X_train[subset_a],
                  y=y_train,
                  validation_split=0.13,
                  epochs=500,
                  batch_size=1000,
                  workers=-1,
                  use_multiprocessing=True,
                  verbose=0,
                  callbacks=[es, mc])

# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
# plot validation accuracy per epoch
plt.figure(figsize=(8,6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.tight_layout()
plt.savefig("net_acc_a_u")
# plot loss per epoch
plt.figure(figsize=(8,6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.tight_layout()
plt.savefig("net_loss_a_u")

# predictions with saved weights from best validation accuracy
mod.load_weights('best_model.h5')
predictions = (mod.predict(X_test[subset_a]) > 0.5).astype("int32")
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(roc_auc_score(y_test, predictions))

# Shapley Values
explainer = shap.DeepExplainer(mod, np.array(X_train[subset_a][:5000]))
shap_values = explainer.shap_values(np.array(X_test[subset_a][:1000]))

# plot mean absolute value
shap_df = pd.DataFrame(shap_values[0],columns=X_train[subset_a].columns)
shap_abs_mean = shap_df.abs().mean().sort_values()
plt.figure(figsize=(8,6))
plt.barh(shap_abs_mean.index, shap_abs_mean)
plt.xlabel("mean |SHAP value|")
plt.grid(False,axis='y')
plt.tight_layout()
plt.savefig("net_shap_ma_a_u")

# plot all values
shap_df = shap_df.melt()
shap_df["sign"] = shap_df["value"] > 0
plt.figure(figsize=(5,10))
ax = sns.stripplot(x=shap_df[shap_df["sign"] == True]["value"],
                   y=shap_df[shap_df["sign"] == True]["variable"],
                   color="red")
ax = sns.stripplot(x=shap_df[shap_df["sign"] != True]["value"],
                   y=shap_df[shap_df["sign"] != True]["variable"],
                   color="blue")
plt.xlabel("SHAP value")
plt.tight_layout()
plt.savefig("net_shap_a_u")


# SUBSET B
print("NEURAL NET SUBSET B")

# build neural net
mod = Sequential()
# input layer
mod.add(Dense(units=13, input_dim=X_train[subset_b].shape[1], activation='relu'))
# first hidden layer
mod.add(Dense(units=9, activation='relu'))
# output layer
mod.add(Dense(units=1, activation='sigmoid'))

mod.compile(loss='binary_crossentropy'
            , optimizer='adam'
            , metrics=['accuracy'])

# https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

# fit neural net
history = mod.fit(x=X_train[subset_b],
                  y=y_train,
                  validation_split=0.13,
                  epochs=500,
                  batch_size=1000,
                  workers=-1,
                  use_multiprocessing=True,
                  verbose=0,
                  callbacks=[es, mc])

# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
# plot validation accuracy per epoch
plt.figure(figsize=(8,6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.tight_layout()
plt.savefig("net_acc_b_u")
# plot loss per epoch
plt.figure(figsize=(8,6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.tight_layout()
plt.savefig("net_loss_b_u")

# predictions with saved weights from best validation accuracy
mod.load_weights('best_model.h5')
predictions = (mod.predict(X_test[subset_b]) > 0.5).astype("int32")
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(roc_auc_score(y_test, predictions))

# Shapley Values
explainer = shap.DeepExplainer(mod, np.array(X_train[subset_b][:5000]))
shap_values = explainer.shap_values(np.array(X_test[subset_b][:1000]))

# plot mean absolute value
shap_df = pd.DataFrame(shap_values[0],columns=X_train[subset_b].columns)
shap_abs_mean = shap_df.abs().mean().sort_values()
plt.figure(figsize=(8,6))
plt.barh(shap_abs_mean.index, shap_abs_mean)
plt.xlabel("mean |SHAP value|")
plt.grid(False,axis='y')
plt.tight_layout()
plt.savefig("net_shap_ma_b_u")

# plot all values
shap_df = shap_df.melt()
shap_df["sign"] = shap_df["value"] > 0
plt.figure(figsize=(5,10))
ax = sns.stripplot(x=shap_df[shap_df["sign"] == True]["value"],
                   y=shap_df[shap_df["sign"] == True]["variable"],
                   color="red")
ax = sns.stripplot(x=shap_df[shap_df["sign"] != True]["value"],
                   y=shap_df[shap_df["sign"] != True]["variable"],
                   color="blue")
plt.xlabel("SHAP value")
plt.tight_layout()
plt.savefig("net_shap_b_u")