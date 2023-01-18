"""
====================
Dice Cofficient Metric
====================

.. math:: -2\\frac{|\mathcal{M} \otimes \hat{\mathcal{M}}|}{|\mathcal{M}| - |\hat{\mathcal{M}}|}

 
.. [1] `Random Fourier Features-Based Deep Learning Improvement with Class Activation Interpretability for Nerve Structure Segmentation`_

.. [2] `Multi categorical Dice loss?`_

.. [3] `RFF-Nerve-UTP`_


.. _`Random Fourier Features-Based Deep Learning Improvement with Class Activation Interpretability for Nerve Structure Segmentation`: http://www.sdss.org/dr14/help/glossary/#stripe

.. _`Multi categorical Dice loss?`: https://stats.stackexchange.com/questions/285640/multi-categorical-dice-loss

.. _`RFF-Nerve-UTP`: https://github.com/cralji/RFF-Nerve-UTP
"""

from tensorflow.keras.metrics import Metric
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
import tensorflow as tf 


class DiceCoefficientMetric(Metric):

    def __init__(self,smooth=1.0, target_class=None, name='DiceCoefficientMetric',**kwargs):
        super().__init__(name=name,**kwargs)
        self.smooth = smooth
        self.target_class = target_class
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")
        

    def update_state(self, y_true, y_pred, sample_weight=None):
        metric = self.compute(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
    
    def result(self):
        return self.total/self.count 

    def compute(self, y_true, y_pred):
        intersection = K.sum(y_true * y_pred, axis=[1,2])
        union = K.sum(y_true,axis=[1,2]) + K.sum(y_pred,axis=[1,2])
        dice_coef = -(2. * intersection + self.smooth) /(union + self.smooth)
        
        if self.target_class != None:
            dice_coef = tf.gather(dice_coef,
                                  self.target_class, axis=1)
        else:
            dice_coef = K.mean(dice_coef,axis=-1)

        return dice_coef
    
    def get_config(self,):
        base_config = super().get_config()
        return {**base_config, "smooth": self.smooth,
                "target_class":self.target_class}


class SparseCategoricalDiceCoefficientMetric(DiceCoefficientMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = to_categorical(y_true)
        return super().update_state(y_true, y_pred)