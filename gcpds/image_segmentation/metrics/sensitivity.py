"""
https://github.com/cralji/RFF-Nerve-UTP
"""

from tensorflow.keras.metrics import Metric
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
import tensorflow as tf 


class Sensitivity(Metric):

    def __init__(self,**kwargs):
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")
        super().__init__(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        metric = self.sensitivity(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
    
    def result(self):
        return self.total/self.count 

    @staticmethod
    def sensitivity(y_true, y_pred):
        y_true = tf.cast(y_true > 0.5,tf.float32)
        y_pred = tf.cast(y_pred > 0.5 ,tf.float32)
    
        true_positves = K.sum(y_true*y_pred,axis=[1,2,3])
        total_positives = K.sum(y_true,axis=[1,2,3])
   
        return tf.reduce_mean(true_positves / (total_positives + K.epsilon()))
    
    def get_config(self,):
        base_config = super().get_config()
        return base_config


class SparseCategoricalSensitivity(Sensitivity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = to_categorical(y_true)
        return super().update_state(y_true, y_pred)