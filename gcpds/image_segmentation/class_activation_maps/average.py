"""
Average drop and average inclrease metrics
"""


import numpy as np 
import tensorflow as tf 


class AveragesCam:
    def __init__(self, model: tf.keras.Model , inputs: tf.Tensor) -> None:
        self.model = model 
        self.inputs = inputs 
        self.Y_c = model.predict(inputs)
    
    def __get_scores(self, cams: tf.Tensor, score_function: 'score function') -> tuple:
        cams = cams[...,None]
        inputs = self.inputs*cams
        O_c = self.model.predict(inputs)
        O_c = score_function(O_c)
        Y_c = score_function(self.Y_c)
        return Y_c, O_c

    def average_drop(self, cams: tf.Tensor, score_function: 'score function') -> float:
        Y_c, O_c = self.__get_scores(cams, score_function)
        return np.mean(np.maximum(0,(Y_c-O_c))/Y_c)*100

    def average_increase(self, cams: tf.Tensor, score_function: 'score function') -> float:
        Y_c, O_c = self.__get_scores(cams, score_function)
        return 100*np.mean(Y_c < O_c)