from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.datasets import imdb
from tensorflow.keras.datasets import reuters
from tensorflow.keras.datasets import mnist


def demo():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print(X_train.shape,y_test.shape)
    (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')
    print(X_train.shape,y_test.shape)

    (X_train, y_train), (X_test, y_test) = imdb.load_data(path="imdb_full.pkl",
                                                      nb_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      test_split=0.1)
    print(X_train.shape,y_test.shape)


    (X_train, y_train), (X_test, y_test) = reuters.load_data(path="reuters.pkl",
                                                             nb_words=None,
                                                             skip_top=0,
                                                             maxlen=None,
                                                             test_split=0.2,
                                                             seed=113,
                                                             start_char=1,
                                                             oov_char=2,
                                                             index_from=3)
    print(X_train.shape,y_test.shape)


    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print(X_train.shape,y_test.shape)

