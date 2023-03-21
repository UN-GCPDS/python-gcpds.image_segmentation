"""
====================
Generalized Cross-Entropy
====================
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
