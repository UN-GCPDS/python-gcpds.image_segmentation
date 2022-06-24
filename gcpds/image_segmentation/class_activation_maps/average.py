"""
Average drop and average increase metrics
"""

from typing import Callable

import numpy as np
import tensorflow as tf
from tensorflow.keras.backend import epsilon


class AveragesCam:
    """ 
    Set of metrics to measure influence of CAM in the ouput model score.
    
    Attributes
    ----------
    model : tensorflow.keras.Model
        Already trained model.
    inputs : tf.Tensor
        Instances to predict scores.
    """

    def __init__(self, model: tf.keras.Model , inputs: tf.Tensor) -> None:
        """ 
        Parameters
        ----------
        model : 
            Already trained model.
        inputs : 
            Instances to predict scores.

        """
        self.model = model 
        self.inputs = inputs 
        self.Y_c = model.predict(inputs)
    
    def _get_scores(self, cams: tf.Tensor, score_function: Callable) -> tuple:
        """ Calculate scores over inputs and masked inputs with cams.

        Parameters
        ----------
        cams :
            Normalized ouput of CAM methods using the model and the inputs
        score_function :
            Same score function used to calculate the CAMs in tf-keras-vis
        
        Returns
        -------
        tuple 
            The fisrt is the scores of the model over the inputs and the second
            is the scores over the masked input with the cams.

        """
        cams = cams[...,None]
        inputs = self.inputs*cams
        O_c = self.model.predict(inputs)
        O_c = np.array(score_function(O_c))
        Y_c = np.array(score_function(self.Y_c))
        return Y_c, O_c

    def average_drop(self, cams: tf.Tensor, score_function: Callable) -> float:
        """ Calculate average drop.

        Parameters
        ----------
        cams :
            Normalized ouput of CAM methods using the model and the inputs
        score_function :
            Same score function used to calculate the CAMs in tf-keras-vis
        
        Returns
        -------
        float 
            average drop

        Note
        ----
        The equation used is 
        .. math::
            100\\frac{1}{N} \sum_{i=1}^N \\frac{max(0,Y_i^c - O_i^c)}{Y_i^c}

        """
        Y_c, O_c = self._get_scores(cams, score_function)
        return np.mean(np.maximum(0,(Y_c-O_c))/(Y_c+epsilon()))*100

    def average_increase(self, cams: tf.Tensor, score_function: Callable) -> float:
        """ Calculate average increase.

        Parameters
        ----------
        cams :
            Normalized ouput of CAM methods using the model and the inputs
        score_function :
            Same score function used to calculate the CAMs in tf-keras-vis
        
        Returns
        -------
        float 
            average increase

        Note
        ----
        The equation used is 
        .. math::
            100\\frac{1}{N} \sum_{i=1}^N Sign(Y_i^c < O_i^c)

        """
        Y_c, O_c = self._get_scores(cams, score_function)
        return 100*np.mean(Y_c < O_c)

