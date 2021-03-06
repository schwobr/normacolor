"""
Model architecture of autoencoder.

**********************************
"""
# coding: utf8
from keras.layers import (Conv2D, BatchNormalization,
                          UpSampling2D, Lambda, Input)
from keras.models import Model
from keras.regularizers import Regularizer
import keras.backend as K


class KL(Regularizer):
    """
    Regularize applying a Kullback-Leibler divergence constraint on
    activations.
    """

    def __init__(self, rho=5e-2):
        self.rho = K.cast_to_floatx(rho)

    def __call__(self, act):
        rho_hat = K.flatten(K.mean(act, axis=0))
        pen = K.sum(self.rho * K.log(self.rho / rho_hat) +
                    (1 - self.rho) * K.log((1 - self.rho) / (1 - rho_hat)))
        return pen

    def get_config(self):
        return {'rho': float(self.rho)}


# first model
class Autoencoder:
    """
    Convolutional autoencoder architecture.

    No maxpooling, use conv2D with stride 2 instead.
    """

    def __init__(self, width=16, depth=3, rho=5e-2):
        """
        Architecture settings.

        Arguments:
            - width: int, first layer number of convolution filters.
            - depth: int, number of convolution layer in the network.

        """
        self.width = width
        self.depth = depth
        self.rho = rho

    def __call__(self, X):
        """
        Call classifier layers on the inputs.

        *************************************
        """
        # encoding
        for k in range(self.depth):
            if k == 0:
                # filtering
                if k < self.depth - 1:
                    Y = Conv2D(self.width * (2 ** k), 3,
                               padding="same", activation="relu")(X)
                    Y = BatchNormalization()(Y)
                else:
                    Y = Conv2D(self.width * (2 ** k), 3,
                               padding="same", activation="sigmoid",
                               activity_regularizer=KL(self.rho))(X)
            else:
                # size / 2
                Y = Conv2D(self.width * (2 ** k), 3, strides=2,
                           padding="same", activation="relu")(Y)
                # filtering
                if k < self.depth - 1:
                    Y = Conv2D(self.width * (2 ** k), 3,
                               padding="same", activation="relu")(Y)
                    Y = BatchNormalization()(Y)
                else:
                    Y = Conv2D(self.width * (2 ** k), 3,
                               padding="same", activation="sigmoid",
                               activity_regularizer=KL(self.rho))(Y)

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


def make_autoencoder_model(width=16, depth=3, patch_size=16):
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
