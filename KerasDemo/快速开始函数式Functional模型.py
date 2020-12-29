# from tensorflow.keras.layers import Input, Dense
# from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense,Conv2D
from tensorflow.keras.models import Model
import tensorflow

def demo():
    # This returns a tensor
    inputs = Input(shape=(784,))

    # a layer instance is callable on a tensor, and returns a tensor
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # model.fit(data, labels)  # starts training


# 多输入和多输出模型

def multi_input_output():
    """
    :return:
    """
    # Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
    # Note that we can name any layer by passing it a "name" argument.
    main_input = Input(shape=(100,), dtype='int32', name='main_input')

    # This embedding layer will encode the input sequence
    # into a sequence of dense 512-dimensional vectors.
    x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

    # A LSTM will transform the vector sequence into a single vector,
    # containing information about the entire sequence
    lstm_out = LSTM(32)(x)
    auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)

    auxiliary_input = Input(shape=(5,), name='aux_input')
    x = tensorflow.keras.layers.concatenate([lstm_out, auxiliary_input])

    # We stack a deep densely-connected network on top
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    # And finally we add the main logistic regression layer
    main_output = Dense(1, activation='sigmoid', name='main_output')(x)
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
    print(model)
    # model.compile(optimizer='rmsprop', loss='binary_crossentropy',
    #               loss_weights=[1., 0.2])
    # model.fit([headline_data, additional_data], [labels, labels],
    #           epochs=50, batch_size=32)
    #
    # model.compile(optimizer='rmsprop',
    #               loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
    #               loss_weights={'main_output': 1., 'aux_output': 0.2})
    #
    # # And trained it via:
    # model.fit({'main_input': headline_data, 'aux_input': additional_data},
    #           {'main_output': labels, 'aux_output': labels},
    #           epochs=50, batch_size=32)


def share_layer():
    """

    :return:
    """
    # import keras
    # from keras.layers import Input, LSTM, Dense
    # from keras.models import Model

    tweet_a = Input(shape=(140, 256))
    tweet_b = Input(shape=(140, 256))

    # This layer can take as input a matrix
    # and will return a vector of size 64
    shared_lstm = LSTM(64)

    # When we reuse the same layer instance
    # multiple times, the weights of the layer
    # are also being reused
    # (it is effectively *the same* layer)
    encoded_a = shared_lstm(tweet_a)
    encoded_b = shared_lstm(tweet_b)

    # We can then concatenate the two vectors:
    merged_vector = tensorflow.keras.layers.concatenate([encoded_a, encoded_b], axis=-1)

    # And add a logistic regression on top
    predictions = Dense(1, activation='sigmoid')(merged_vector)

    # We define a trainable model linking the
    # tweet inputs to the predictions
    model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    # model.fit([data_a, data_b], labels, epochs=10)


def output_type():
    a = Input(shape=(140, 256))

    lstm = LSTM(32)
    encoded_a = lstm(a)

    print(lstm.output)
    print(encoded_a)

def output_type2():
    a = Input(shape=(32, 32, 3))
    b = Input(shape=(64, 64, 3))

    conv = Conv2D(16, (3, 3), padding='same')
    conved_a = conv(a)

    # Only one input so far, the following will work:
    assert conv.input_shape == (None, 32, 32, 3)

    conved_b = conv(b)
    # now the `.input_shape` property wouldn't work, but this does:
    assert conv.get_input_shape_at(0) == (None, 32, 32, 3)
    assert conv.get_input_shape_at(1) == (None, 64, 64, 3)


if __name__=='__main__':
    output_type2()