from tensorflow.keras.layers import Dense

def get_config():
    layer = Dense(32)
    config = layer.get_config()
    reconstructed_layer = Dense.from_config(config)
    print(reconstructed_layer)

    print(layer.input)
    print(layer.output)
    print(layer.input_shape)
    print(layer.output_shape)

    print(layer.get_input_at)
    print(layer.get_output_at)
    print(layer.get_input_shape_at)
    print(layer.get_output_shape_at)


"""
看不懂的层：
SeparableConv2D
Conv2DTranspose
Masking
Embedding
包装器Wrapper
"""