"""
====================
Jaccard Metric
====================

.. math:: \\frac{|\mathcal{M} \cap \hat{\mathcal{M}}|}{|\mathcal{M} \cup \hat{\mathcal{M}}|}

 
.. [1] `Random Fourier Features-Based Deep Learning Improvement with Class Activation Interpretability for Nerve Structure Segmentation`_

.. [2] `RFF-Nerve-UTP`_

.. _`Random Fourier Features-Based Deep Learning Improvement with Class Activation Interpretability for Nerve Structure Segmentation`: http://www.sdss.org/dr14/help/glossary/#stripe

.. _`RFF-Nerve-UTP`: https://github.com/cralji/RFF-Nerve-UTP
"""

from tensorflow.keras.metrics import Metric
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
import tensorflow as tf 


class Jaccard(Metric):

    def __init__(self,smooth=1.0, target_class=None, name='Jaccard',**kwargs):
        super().__init__(name=name, **kwargs)
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
        union = K.sum(y_true, axis=[1,2]) + K.sum(y_pred, axis=[1,2])
        jaccard = (intersection + self.smooth) / (union - intersection + self.smooth)

        if self.target_class != None:
            jaccard = tf.gather(jaccard, 
                                self.target_class, axis=1)
        else: 
            jaccard = K.mean(jaccard,axis=-1)
            
        return jaccard

    
    def get_config(self,):
        base_config = super().get_config()
        return {**base_config, "smooth": self.smooth,
                "target_class":self.target_class}


class SparseCategoricalJaccard(Jaccard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = to_categorical(y_true)
        return super().update_state(y_true, y_pred)