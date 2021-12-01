import pandas as pd
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import datasets, layers, models
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import matplotlib.pyplot as plt
import sklearn

def traffic_model():
    train_data = pd.read_csv("data/train.csv").fillna(0)
    train_data =train_data.sample(frac=1)
    test_data = pd.read_csv("data/test.csv").fillna(0)
    test_data =test_data.sample(frac=1)
    merged_data = pd.concat(
        (train_data, test_data),
        axis=0,
        join="outer")
    merged_data.head()
    train_data = merged_data.sample(frac=0.8, random_state=20)
    test_data = merged_data.drop(train_data.index)
    attribute = 'Lane 1 Flow (Veh/5 Minutes)'
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(train_data[attribute].values.reshape(-1, 1))
    train_transform = scaler.transform(train_data[attribute].values.reshape(-1, 1)).reshape(1, -1)[0]
    test_transform = scaler.transform(test_data[attribute].values.reshape(-1, 1)).reshape(1, -1)[0]
    train, test = [], []
    delay = 256
    for i in range(delay, len(train_transform)):
        train.append(train_transform[i - delay: i + 1])
    for i in range(delay, len(test_transform)):
        test.append(test_transform[i - delay: i + 1])
    train = np.array(train)
    test = np.array(test)
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]
    y_train
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], -1))
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], -1))
    X_train.shape
    X_test.shape
    X_train = tf.transpose(X_train, [0, 2, 1])
    X_test = tf.transpose(X_test, [0, 2, 1])
    model = models.Sequential()
    model = models.Sequential()
    model.add(layers.Conv1D(256, 1, activation='softmax', input_shape=(1, 256)))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(layers.Conv1D(64, 1, activation='softmax',  input_shape=(256, 64)))
    model.add(layers.Conv1D(64, 1, activation='softmax',  input_shape=(128, 16)))
    model.add(layers.Conv1D(64, 1, activation='softmax',  input_shape=(64, 8)))
    model.add(layers.Dropout(.2, input_shape=(4,8)))
    model.add(layers.MaxPooling1D(1))
    model.add(layers.Conv1D(64, 1, activation='relu'))
    # model.summary()
    model.compile(optimizer='Adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ["categorical_accuracy"])
    history = model.fit(X_train, y_train, batch_size = 32, epochs=30, shuffle=True,verbose=1,
                        validation_data=(X_test, y_test))
def predict(dateandtime):
    loaded_model = pickle.load(open('model', 'rb'))
    prediction = loaded_model.predict(dateandtime)



