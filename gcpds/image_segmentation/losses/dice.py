"""
https://github.com/cralji/RFF-Nerve-UTP
https://stats.stackexchange.com/questions/285640/multi-categorical-dice-loss
"""

from tensorflow.keras.losses import Loss
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

class DiceCoeficiente(Loss):
    def __init__(self, smooth=1., **kwargs):
        self.smooth = smooth
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        intersection = K.sum(y_true * y_pred)
        return -(2. * intersection + self.smooth) / (K.sum(y_true) + K.sum(y_pred) + self.smooth)

    def get_config(self,):
        base_config = super().get_config()
        return {**base_config, "smooth": self.smooth}


class SparseCategoricalDiceCoeficiente(DiceCoeficiente):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        y_true = to_categorical(y_true)
        return super().call(y_true, y_pred)
