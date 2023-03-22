"""
====================
Generalized Cross-Entropy
====================


.. math:: \\frac{1}{H*W}\sum_i^H \sum_j^W 2\\frac{1- (\sum_k^C\mathbf{M}_{i,j,k}* \hat{\mathbf{M}}_{i,j,k})^q}{q}

 
.. [1] `Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels`_

.. _`Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels`: https://proceedings.neurips.cc/paper_files/paper/2018/file/f2925f97bc13ad2852a7a551802feea0-Paper.pdf

"""

from tensorflow.keras.losses import Loss
from tensorflow.keras import backend as K

class GeneralizedCrossEntropy(Loss):
    def __init__(self, q=1, name='GeneralizedCrossEntropy',**kwargs):
        self.q = q
        super().__init__(name=name,**kwargs)

    def call(self, y_true, y_pred):
        haddamard = y_true*y_pred
        haddamard_sum = K.sum(haddamard, axis=[-1]) # batch_size, h, w
        results = 2 *(1 - K.pow(haddamard_sum,self.q))/self.q
        results = K.mean(results, axis=[1,2]) # bath_size, value 
        return results 
    
    def get_config(self,):
        base_config = super().get_config()
        return {**base_config, "q": self.q}

if __name__ == "__main__":
    import numpy as np 
    loss = GeneralizedCrossEntropy()
    Y = np.ones(shape=(20,20,20,3))
    Y_pred = Y.copy()
    Y_pred[0,...]=0
    print(loss(Y,Y_pred))
