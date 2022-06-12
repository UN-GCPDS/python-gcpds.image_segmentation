"""
Score for semantic segmentation models to use tf-keras-vis.
"""


import tensorflow as tf
import matplotlib.pyplot as plt 


class SegScore:
    """
    Score for semantic segmentation models to use tf-keras-vis.
    """
    def __init__(self, target_mask: tf.Tensor, class_channel: int = 0) -> None:
        self.target_mask = self.__sparse(target_mask)
        self.class_channel = class_channel

    def __call__(self, pred: tf.Tensor) -> tf.Tensor:
        channels = pred.shape[-1]

        class_mask = self.target_mask == self.class_channel
        class_mask = tf.cast(class_mask,tf.float32)

        class_pred = pred
        if channels != 1:
            class_pred = class_pred[...,self.class_channel]
            class_pred = class_pred[...,None]
        else: 
            if self.class_channel == 0:
                class_pred = 1-class_pred
        
        masked_scores = class_mask*class_pred
        mean_scores = tf.reduce_mean(masked_scores, axis=[-1,-2,-3])
        return mean_scores

    @staticmethod
    def __sparse(masks: tf.Tensor) -> tf.Tensor:
        channels = masks.shape[-1]
        if channels == 1:
            return masks 
        sparse_mask = tf.argmax(masks,axis=-1)
        return sparse_mask[...,None]