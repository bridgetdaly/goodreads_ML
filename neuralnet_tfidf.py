import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
import shap
import seaborn as sns
sns.set_theme(style="whitegrid")

# read data
X_train = pd.read_pickle("data/X_train.pkl")
X_test = pd.read_pickle("data/X_test.pkl")
y_train = pd.read_pickle("data/y_train.pkl")
y_test = pd.read_pickle("data/y_test.pkl")

# TF-IDF
print("NEURAL NET TF-IDF")

# build tfidf
tf = ColumnTransformer(remainder='passthrough',
                       transformers=[('tfidfvectorizer',
                                      TfidfVectorizer(min_df=0.0001),
                                      'tokenized_words')])
X_train_tf = tf.fit_transform(X_train).toarray()
X_test_tf = tf.transform(X_test).toarray()

# build neural net
mod = Sequential()
# input layer
mod.add(Dense(units=26115, input_dim=X_train_tf.shape[1], activation='relu'))
# first hidden layer
mod.add(Dense(units=17411, activation='relu'))
# output layer
mod.add(Dense(units=1, activation='sigmoid'))

mod.compile(loss='binary_crossentropy'
            , optimizer='adam'
            , metrics=['accuracy'])

# https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

# fit neural net
history = mod.fit(x=X_train_tf,
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
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("net_acc_tf")
# plot loss per epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("net_loss_tf")

# predictions with saved weights from best validation accuracy
mod.load_weights('best_model.h5')
predictions = (mod.predict(X_test_tf) > 0.5).astype("int32")
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(roc_auc_score(y_test, predictions))

# Shapley Values
explainer = shap.DeepExplainer(mod, np.array(X_train_tf[:5000]))
shap_values = explainer.shap_values(np.array(X_test_tf[:1000]))

# plot mean absolute value
shap_df = pd.DataFrame(shap_values[0],columns=tf.get_feature_names())
shap_abs_mean = shap_df.abs().mean().sort_values()
plt.figure(figsize=(8,6))
plt.barh(shap_abs_mean.index, shap_abs_mean)
plt.xlabel("mean |SHAP value|")
plt.grid(False,axis='y')
plt.tight_layout()
plt.savefig("net_shap_ma_tf")

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
plt.savefig("net_shap_tf")