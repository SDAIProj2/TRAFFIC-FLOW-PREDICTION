from tensorflow.keras.layers import Dense, Dropout, Activation
# from keras.layers.recurrent import GRU
from tensorflow.keras.models import Sequential


def gru(units):

    model1 = models.Sequential()
    model1 = models.Sequential()
    model1.add(layers.Conv1D(256, 3, activation='relu', input_shape=(256, 256)))
    model1.add(layers.Conv1D(128,3, activation='relu', input_shape=(128, 128)))
    model1.add(layers.Conv1D(64, 3, activation='relu'))
    model1.add(layers.MaxPooling1D(1))
    model1.add(layers.Conv1D(64, 3, activation='relu'))

    return model
