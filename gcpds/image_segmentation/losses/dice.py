"""
* https://github.com/cralji/RFF-Nerve-UTP
* https://stats.stackexchange.com/questions/285640/multi-categorical-dice-loss
* "It is also preferable to return a tensor containing one loss per instance, rather
than returning the mean loss. This way, Keras can apply class weights or sample
weights when requested" - Hands on machine learning with Scikit-Learn and TensorFlow
"""

from tensorflow.keras.losses import Loss
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

class DiceCoeficiente(Loss):
    def __init__(self, smooth=1., target_class= None, name='DiceCoeficiente', **kwargs):
        self.smooth = smooth
        self.target_class = target_class
        super().__init__(name=name,**kwargs)

    def call(self, y_true, y_pred):
        intersection = K.sum(y_true * y_pred, axis=[1,2])
        union = K.sum(y_true,axis=[1,2]) + K.sum(y_pred,axis=[1,2])
        dice_coef_per_class = -(2. * intersection + self.smooth) /(union + self.smooth)

        if self.target_class != None:
            dice_coef_per_class = tf.gather(dice_coef_per_class,
                                             self.target_class, axis=1)

        dice_coef_mean_per_class = K.mean(dice_coef_per_class,axis=-1)
        return dice_coef_mean_per_class

    def get_config(self,):
        base_config = super().get_config()
        return {**base_config, "smooth": self.smooth,
                "target_class":self.target_class}


class SparseCategoricalDiceCoeficiente(DiceCoeficiente):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        y_true = to_categorical(y_true)
        return super().call(y_true, y_pred)
