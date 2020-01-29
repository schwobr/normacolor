"""
Model architecture of autoencoder.

**********************************
"""
# coding: utf8
from keras.layers import (Conv2D, BatchNormalization,
                          UpSampling2D, Lambda, Input)
from keras.models import Model


# first model
class Autoencoder:
    """
    Convolutional autoencoder architecture.

    No maxpooling, use conv2D with stride 2 instead.
    """

    def __init__(self, width=16, depth=3):
        """
        Architecture settings.

        Arguments:
            - width: int, first layer number of convolution filters.
            - depth: int, number of convolution layer in the network.

        """
        self.width = width
        self.depth = depth

    def __call__(self, X):
        """
        Call classifier layers on the inputs.

        *************************************
        """
        # encoding
        for k in range(self.depth):
            if k == 0:
                # filtering
                Y = Conv2D(self.width * (2 ** k), 3,
                           padding="same", activation="relu")(X)
                Y = BatchNormalization()(Y)
            else:
                # size / 2
                Y = Conv2D(self.width * (2 ** k), 3, strides=2,
                           padding="same", activation="relu")(Y)
                # filtering
                Y = Conv2D(self.width * (2 ** k), 3,
                           padding="same", activation="relu")(Y)
                Y = BatchNormalization()(Y)

        Y = Lambda(lambda t: t, name="encoding")(Y)
        # decoding
        for k in range(self.depth - 1):
            actual_depth = self.depth - k - 2
            # size * 2
            Y = UpSampling2D()(Y)
            # filtering
            Y = Conv2D(self.width * (2 ** actual_depth), 3,
                       padding='same', activation='relu')(Y)
            Y = BatchNormalization()(Y)
        Y = Conv2D(3, 1, padding="same", activation="relu")(Y)
        return Y


def make_autoencoder_model(width=16, depth=3, patch_size=15):
    """
    Create and compile autoencoder keras model.

    *******************************************
    """
    X = Input(batch_shape=(None, patch_size, patch_size, 3))
    Y = Autoencoder(width=width, depth=depth)(X)
    model = Model(inputs=X, outputs=Y)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# autoencoder = make_autoencoder_model()
# autoencoder.summary()
