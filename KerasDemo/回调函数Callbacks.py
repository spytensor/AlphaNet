from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
import tensorflow
import sklearn.datasets as dataset


def record_loss():
    class LossHistory(tensorflow.keras.callbacks.Callback):
        def __init__(self):
            super(LossHistory, self).__init__()
            self.losses = []

        def on_train_begin(self, logs=None):
            print('train begin')

        def on_batch_end(self, batch, logs=None):
            self.losses.append(logs.get('loss'))

    model = Sequential()
    model.add(Dense(1, input_dim=4, kernel_initializer='uniform'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    history = LossHistory()
    iris_data = dataset.load_iris()

    X_train = iris_data['data']
    Y_train = iris_data['target']

    model.fit(X_train, Y_train, batch_size=128, epochs=20, verbose=0, callbacks=[history])

    print(history.losses)
    # outputs
    '''
    [0.66047596406559383, 0.3547245744908703, ..., 0.25953155204159617, 0.25901699725311789]
    '''

if __name__=='__main__':
    record_loss()