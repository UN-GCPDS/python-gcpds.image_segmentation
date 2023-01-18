"""
====================
Sensitivity Metric
====================
 
.. [1] `Random Fourier Features-Based Deep Learning Improvement with Class Activation Interpretability for Nerve Structure Segmentation`_

.. [2] `RFF-Nerve-UTP`_

.. _`Random Fourier Features-Based Deep Learning Improvement with Class Activation Interpretability for Nerve Structure Segmentation`: http://www.sdss.org/dr14/help/glossary/#stripe

.. _`RFF-Nerve-UTP`: https://github.com/cralji/RFF-Nerve-UTP
"""

from tensorflow.keras.metrics import Metric
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
import tensorflow as tf 


class Sensitivity(Metric):

    def __init__(self, target_class=None, name='Sensitivity', **kwargs):
        super().__init__(name=name, **kwargs)
        self.target_class = target_class
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        metric = self.sensitivity(y_true, y_pred, self.target_class)
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
    
    def result(self):
        return self.total/self.count 

    def compute(self,y_true, y_pred):
        return self.sensitivity(y_true, y_pred, self.target_class)

    @staticmethod
    def sensitivity(y_true, y_pred, target_class=None):
        y_true = tf.cast(y_true > 0.5,tf.float32)
        y_pred = tf.cast(y_pred > 0.5 ,tf.float32)
    
        true_positves = K.sum(y_true*y_pred,axis=[1,2])
        total_positives = K.sum(y_true,axis=[1,2])

        sensitivity = true_positves / (total_positives + K.epsilon())

        if target_class != None:
            sensitivity = tf.gather(sensitivity,
                                    target_class, axis=1)
        else:
            sensitivity = K.mean(sensitivity,axis=-1)
            
        return sensitivity
    
    def get_config(self,):
        base_config = super().get_config()
        return {**base_config, "target_class":self.target_class}


class SparseCategoricalSensitivity(Sensitivity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = to_categorical(y_true)
        return super().update_state(y_true, y_pred)