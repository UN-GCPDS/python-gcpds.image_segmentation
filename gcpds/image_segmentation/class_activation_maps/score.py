"""
==========================================================
Score for Semantic Segmentation Models to use tf-keras-vis
==========================================================
"""


import tensorflow as tf
from tensorflow.keras.backend import epsilon


class SegScore:
    """
    Score for semantic segmentation models to use tf-keras-vis.

    Attributes
    ----------
    target_mask : tf.Tensor
        Masks of the interest regions
    target_class : int 
        Label or channel of the interest class.

    """

    def __init__(self, target_mask: tf.Tensor, target_class: int = 0, 
                logits: bool = False) -> None:
        """
        Parameters
        ----------
        target_mask : 
            Masks of the interest regions
        target_class : 
            Label or channel of the interest class.
        logits :
            If the values are from logit 
        """
        self.target_mask = self.__sparse(target_mask)
        self.target_class = target_class
        self.logits = logits

    def __call__(self, pred: tf.Tensor) -> tf.Tensor:
        """ Calculate scores 

        Parameters
        ----------
        pred : 
            Predictions (masks) of a tf.keras.Model

        Returns
        -------
        tuple 
            Score per instances 

        """
        channels = pred.shape[-1]

        class_mask = self.target_mask == self.target_class
        class_mask = tf.cast(class_mask,tf.float32)

        class_pred = pred
        if channels != 1:
            class_pred = class_pred[...,self.target_class]
            class_pred = class_pred[...,None]
        else: 
            if self.target_class == 0:
                if self.logits:
                    class_pred = -class_pred
                elif not self.logits:
                    class_pred = 1-class_pred
        
        masked_scores = class_mask*class_pred
        N = tf.reduce_sum(class_mask,axis=[-1,-2,-3])
        sum_scores = tf.reduce_sum(masked_scores, axis=[-1,-2,-3])
        mean_scores = sum_scores/(N+epsilon())
        return mean_scores

    @staticmethod
    def __sparse(masks: tf.Tensor) -> tf.Tensor:
        """ Ensures target class be sparse 

        """
        channels = masks.shape[-1]
        if channels == 1:
            return masks 
        sparse_mask = tf.argmax(masks,axis=-1)
        return sparse_mask[...,None]