from typing import Callable, Iterable

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from .score import SegScore


class FusionCam:
    def __init__(self, callable_cam : Callable, dataset : tf.data, layers : Iterable, 
                 target_class: int,  logits: bool = False):
        self.callable_cam = callable_cam
        self.dataset = dataset
        self.layers = layers 
        self.target_class = target_class 
        self.logits = logits 
        self.weights = self._get_weigths()


    @staticmethod
    def normalize_weigths(weights):
        normalize_weights = weights/np.max(weights)
        normalize_weights = tf.math.softmax(normalize_weights)
        return normalize_weights


    def _get_weigths(self,):
        weights = np.zeros(shape=(len(self.layers)))

        layers = tqdm(self.layers)
        for i,layer in enumerate(layers):
            layers.set_postfix({'Layer ':layer})
            values = []
            dataset = tqdm(self.dataset)
            for j,(img,mask) in enumerate(dataset):
                dataset.set_postfix({'Bacth ': j+1})
                cam = self.callable_cam(SegScore(mask,
                                                 target_class=self.target_class,
                                                 logits=self.logits), 
                                        img,
                                        penultimate_layer=layer,
                                        seek_penultimate_conv_layer=False,
                                        normalize_cam =False)
                value = tf.reduce_sum(cam[...,None]*mask, axis=[1,2])/tf.reduce_sum(mask,axis=[1,2])
                values.extend(value)

            values = np.array(values)
            weights[i] = np.mean(values)

        return weights
    

    def __call__(self, dataset: tf.data, norm_weights: bool=True):
        weights = FusionCam.normalize_weigths(self.weights) if norm_weights else self.weights

        layers = tqdm(self.layers)
        result = 0 
        for i,layer in enumerate(layers):
            layers.set_postfix({'Layer ':layer})

            dataset_ = tqdm(dataset)
            cams = []
            imgs = []
            for j,(img,mask) in enumerate(dataset_):
                dataset_.set_postfix({'Bacth ': j})
                cam = self.callable_cam(SegScore(mask,
                                                 target_class=self.target_class,
                                                 logits=self.logits), 
                                        img,
                                        penultimate_layer=layer,
                                        seek_penultimate_conv_layer=False)
                cams.append(cam)
                imgs.append(img)          
            result += weights[i]*np.vstack(cams)
        min_ = tf.reduce_min(result,axis=[1,2],keepdims=True)
        max_ = tf.reduce_max(result,axis=[1,2],keepdims=True)
        result = (result  - min_) /(max_ - min_)

        return np.array(result),np.vstack(imgs)