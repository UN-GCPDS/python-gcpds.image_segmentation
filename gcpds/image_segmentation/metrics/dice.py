"""
https://github.com/cralji/RFF-Nerve-UTP
"""

from tensorflow.keras.metrics import Metric
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
import tensorflow as tf 


class DiceCoeficiente(Metric):

    def __init__(self,smooth=1.0,name='DiceCoeficiente',**kwargs):
        super().__init__(name='DiceCoeficiente',**kwargs)
        self.smooth = smooth
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")
        

    def update_state(self, y_true, y_pred, sample_weight=None):
        metric = self.dice(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
    
    def result(self):
        return self.total/self.count 

    def dice(self, y_true, y_pred):
        intersection = K.sum(y_true * y_pred, axis=[1,2,3])
        union = K.sum(y_true,axis=[1,2,3]) + K.sum(y_pred,axis=[1,2,3])
        return (2. * intersection + self.smooth) /(union + self.smooth)
    
    def get_config(self,):
        base_config = super().get_config()
        return {**base_config, "smooth": self.smooth}


class SparseCategoricalDiceCoeficiente(DiceCoeficiente):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = to_categorical(y_true)
        return super().update_state(y_true, y_pred)