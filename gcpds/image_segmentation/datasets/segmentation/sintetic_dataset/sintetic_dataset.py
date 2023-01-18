"""
================
Sintetic Dataset
================
"""
import random

import numpy as np
import skimage
import tensorflow as tf

from .texture_generation import NoiseUtils


class SinteticDataset: 
    def __init__(self, seed=42, samples=10, img_shape=128,
                 blob_size_fraction=0.3, target_texture='cloud'):
        self.seed = seed 
        self.samples = samples 
        self.img_shape = img_shape
        self.blob_size_fraction = blob_size_fraction
        self.target_texture = target_texture
        self.textures = ['marble','wood','cloud']
        self.textures.remove(target_texture)
    
    def __generate_structure(self, texture):
        structure = NoiseUtils(self.img_shape)
        texture = getattr(structure,texture)
        structure.makeTexture(texture = texture)
        structure = structure.img 
        return structure

    def generate_sample(self, seed):
        img = self.__generate_structure(self.target_texture)
        mask = org_mask = skimage.data.binary_blobs(length=self.img_shape,
                                         blob_size_fraction=self.blob_size_fraction,
                                         seed=seed)
        img = img*mask
        
        np.random.shuffle(self.textures)
        for i,texture in enumerate(self.textures):
            img1 = self.__generate_structure(texture) 
            img = img + img1*(~mask)
            mask = skimage.data.binary_blobs(length=self.img_shape,
                                         blob_size_fraction=self.blob_size_fraction,
                                         seed=seed+i+1) + org_mask

            if random.uniform(0, 1) > 0.5:
                break 
            
        return img[...,None], org_mask[...,None]

    def gen_dataset(self):
        def generator():
            np.random.seed(self.seed)
            seeds = np.random.randint(self.samples,size=self.samples)
            for seed in seeds:
                img, mask = self.generate_sample(seed)
                yield img, mask 
        return generator

    def __call__(self,):
        output_signature = (tf.TensorSpec((None,None,None), tf.float32), 
                            tf.TensorSpec((None,None,None), tf.float32))
        return tf.data.Dataset.from_generator(self.gen_dataset(),
                                    output_signature = output_signature
                                    )
