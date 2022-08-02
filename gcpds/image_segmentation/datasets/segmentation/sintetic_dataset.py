import tensorflow as tf 
import numpy as np 
import skimage

class SinteticDataset: 
    def __init__(self, seed=42, samples=10, img_shape=128,
                 blob_size_fraction=0.3):
        self.seed = seed 
        self.samples = samples 
        self.img_shape = img_shape
        self.blob_size_fraction = blob_size_fraction

    def generate_sample(self,seed):
        mask = skimage.data.binary_blobs(length=self.img_shape,
                                         blob_size_fraction=self.blob_size_fraction,
                                         seed=seed)

        np.random.seed(seed)
        img1 = mask*(np.random.uniform(0,1,size=(self.img_shape,self.img_shape)))

        np.random.seed(seed)
        img2 = ~mask*(np.random.normal(size=(self.img_shape,self.img_shape),
                                       loc=0.5,scale=0.125))

        img =  img2 + img1
        img = (img - img.min())/(img.max()-img.min())
        return img[...,None], mask[...,None]

    def gen_dataset(self):
        def generator():
            np.random.seed(self.seed)
            seeds = np.random.randint(self.samples,size=self.samples)
            for seed in seeds:
                img, mask = self.generate_sample(seed)
                yield img, mask 
        return generator

    def __call__(self,):
        return tf.data.Dataset.from_generator(self.gen_dataset(),
                                    output_signature = (tf.TensorSpec((None,None,None), tf.float32), 
                                                        tf.TensorSpec((None,None,None), tf.float32)))
