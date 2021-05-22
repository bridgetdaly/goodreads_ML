import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint

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

# BOW
print("NEURAL NET BOW")

# build bag of words
bow = ColumnTransformer(remainder='passthrough',
                        transformers=[('countvectorizer',
                                       CountVectorizer(max_features=1000),
                                       'tokenized_words')])
X_train_bow = bow.fit_transform(X_train).toarray()
X_test_bow = bow.fit_transform(X_test).toarray()

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

# https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

# fit neural net
history = mod.fit(x=X_train_bow,
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