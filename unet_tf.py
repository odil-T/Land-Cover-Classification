import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose, Input, Activation,
                                     concatenate)

def conv_block(input_tensor, n_filters):
    x = Conv2D(n_filters, (3, 3), padding="same")(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(n_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    convolved_tensor = Activation("relu")(x)

    return convolved_tensor


def encoder_block(input_tensor, n_filters):
    convolved_tensor = conv_block(input_tensor, n_filters)
    encoded_tensor = MaxPooling2D((2, 2), strides=2)(convolved_tensor)
    return encoded_tensor, convolved_tensor


def decoder_block(input_tensor, concat_tensor, n_filters):
    x = Conv2DTranspose(n_filters, (2, 2), strides=2, padding="same")(input_tensor)
    x = concatenate([x, concat_tensor], axis=-1)
    decoded_tensor = conv_block(x, n_filters)
    return decoded_tensor


def build_unet(image_patch_size, n_classes):
    input_tensor = Input((image_patch_size, image_patch_size, 3))  # can be 4D for batch_size

    # Encoder
    encoded_tensor, encoders_conv_tensor1 = encoder_block(input_tensor, 64)
    encoded_tensor, encoders_conv_tensor2 = encoder_block(encoded_tensor, 128)
    encoded_tensor, encoders_conv_tensor3 = encoder_block(encoded_tensor, 256)
    encoded_tensor, encoders_conv_tensor4 = encoder_block(encoded_tensor, 512)

    # Bottleneck
    bottleneck_tensor = conv_block(encoded_tensor, 1024)

    # Decoder
    decoded_tensor = decoder_block(bottleneck_tensor, encoders_conv_tensor4, 512)
    decoded_tensor = decoder_block(decoded_tensor, encoders_conv_tensor3, 256)
    decoded_tensor = decoder_block(decoded_tensor, encoders_conv_tensor2, 128)
    decoded_tensor = decoder_block(decoded_tensor, encoders_conv_tensor1, 64)

    # Output
    outputs = Conv2D(n_classes, (1, 1), padding="same", activation="softmax")(decoded_tensor)

    # Model
    model = tf.keras.models.Model(inputs=input_tensor, outputs=outputs)
    print(model.summary())

    return model
