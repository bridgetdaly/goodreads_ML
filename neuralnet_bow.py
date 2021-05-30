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

# BOW
print("NEURAL NET BOW")

# build bag of words
bow = ColumnTransformer(remainder='passthrough',
                        transformers=[('countvectorizer',
                                       CountVectorizer(max_features=10000),
                                       'tokenized_words')])
X_train_bow = bow.fit_transform(X_train)
X_test_bow = bow.transform(X_test)

X_train_bow, X_valid_bow, y_train, y_valid = train_test_split(X_train_bow,
                                                              y_train,
                                                              test_size = 0.13,
                                                              random_state = 229)

# build neural net
mod = Sequential()
# input layer
mod.add(Dense(units=10013, input_dim=X_train_bow.shape[1], activation='relu'))
# first hidden layer
mod.add(Dense(units=6676, activation='relu'))
# output layer
mod.add(Dense(units=1, activation='sigmoid'))

mod.compile(loss='binary_crossentropy'
            , optimizer='adam'
            , metrics=['accuracy'])

# build generator (sparse too big to convert to dense all at once)
batch_size = 1000
epochs = 500
samples_per_epoch = X_train_bow.shape[0]
batches_per_epoch = samples_per_epoch//batch_size

# https://stackoverflow.com/questions/37609892/keras-sparse-matrix-issue
def batch_generator(X_train, y_train, batch_size, batches_per_epoch):
    counter=0
    shuffle_index = np.arange(np.shape(y_train)[0])
    np.random.shuffle(shuffle_index)
    X =  X_train[shuffle_index, :]
    y =  y_train.to_numpy()[shuffle_index]
    while 1:
        index_batch = shuffle_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[index_batch,:].toarray()
        y_batch = y[index_batch]
        counter += 1
        yield(X_batch,y_batch)
        if (counter == batches_per_epoch):
            np.random.shuffle(shuffle_index)
            counter = 0

# https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

# fit neural net
history = mod.fit(x=batch_generator(X_train_bow, y_train, batch_size, batches_per_epoch),
                  validation_data=(X_valid_bow.toarray(), y_valid),
                  epochs=500,
                  steps_per_epoch=batches_per_epoch,
                  workers=-1,
                  use_multiprocessing=True,
                  verbose=1,
                  callbacks=[es, mc])

# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
# plot validation accuracy per epoch
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("net_acc_bow")
# plot loss per epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("net_loss_bow")

# predictions with saved weights from best validation accuracy
mod.load_weights('best_model.h5')
predictions = (mod.predict(X_test_bow) > 0.5).astype("int32")
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(roc_auc_score(y_test, predictions))

# Shapley Values
explainer = shap.DeepExplainer(mod, X_train_bow[:5000].toarray())
shap_values = explainer.shap_values(X_test_bow[:1000].toarray())

# plot mean absolute value
shap_df = pd.DataFrame(shap_values[0],columns=bow.get_feature_names())
shap_abs_mean = shap_df.abs().mean().sort_values()
plt.figure(figsize=(8,6))
plt.barh(shap_abs_mean[::-1][:25].index, shap_abs_mean[::-1][:25])
plt.xlabel("mean |SHAP value|")
plt.grid(False,axis='y')
plt.tight_layout()
plt.savefig("net_shap_ma_bow")

# plot all values
shap_df = shap_df.melt()
shap_df["sign"] = shap_df["value"] > 0
shap_df_plot = shap_df[shap_df["variable"].isin(shap_abs_mean[::-1][:25].index)]
plt.figure(figsize=(5,10))
ax = sns.stripplot(x=shap_df_plot[shap_df_plot["sign"] == True]["value"],
                   y=shap_df_plot[shap_df_plot["sign"] == True]["variable"],
                   color="red")
ax = sns.stripplot(x=shap_df_plot[shap_df_plot["sign"] != True]["value"],
                   y=shap_df_plot[shap_df_plot["sign"] != True]["variable"],
                   color="blue")
plt.xlabel("SHAP value")
plt.tight_layout()
plt.savefig("net_shap_bow")
