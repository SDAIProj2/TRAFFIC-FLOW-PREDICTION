from tensorflow.keras.layers import Dense, Dropout, Activation
# from keras.layers.recurrent import LSTM, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras import models


def get_lstm(units):
    model = Sequential()
    model.add(LSTM(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(LSTM(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model


def get_gru(units):
    model = Sequential()
    model.add(GRU(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(GRU(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))
    return model


def _get_sae(inputs, hidden, output):
    model = Sequential()
    model.add(Dense(hidden, input_dim=inputs, name='hidden'))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(output, activation='sigmoid'))

    return model

def get_cnn(units):
    model = Sequential()
    model.add(layers.Conv1D(32, 2, activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.Conv1D(32, 2, activation='relu', input_shape=(16, 16, 3)))
    model.add(layers.Conv1D(32, 2, activation='relu', input_shape=(8, 8, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    return model

