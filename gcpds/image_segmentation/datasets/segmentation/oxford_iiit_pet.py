"""
I only made this code to keep the interface (NervetUPT, ultrasound_nerve_brachial_plexus)
 (__call__, load_instance_by_id, and order of tuple), 
I still recommend using tfds to create the datasets
"""
import tensorflow as tf
import tensorflow_datasets as tfds
from functools import lru_cache

class OxfordIiitPet:
    def __init__(self, split=[70, 15, 15], seed: int=42, one_hot: bool=True):
        self.one_hot = one_hot
        self.split= OxfordIiitPet._get_splits(split)
        dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True, split=self.split)
        self.info = info 
        self.train, self.val, self.test  = dataset
        self.classes = 3
        self.train = self.train.map(lambda x: self._keep_interface(x)) 
        self.val= self.val.map(lambda x: self._keep_interface(x)) 
        self.test = self.test.map(lambda x: self._keep_interface(x)) 
        self.labels_info = {0:'cat', 1:'dog'}  

    @lru_cache(maxsize=None)
    def load_instance_by_id(self, id_img):
        for dataset in [self.train, self.val, self.test]:
            dataset = dataset.filter(lambda img, mask, label, id_image : id_image ==id_img)
            for x in dataset:
                return x 

    @staticmethod
    def _get_splits(splits):
        sum = 0
        splits_ = []
        for percentage in splits:
            sum += percentage
            splits_.append(f'train[{sum-percentage}%:{sum}%]')
        return splits_

    def to_one_hot(self, mask):
        one_hot = tf.one_hot(mask, self.classes)
        return tf.gather(one_hot,0,axis=2)

    def _keep_interface(self, x):
        img = tf.cast(x['image'], tf.float32)/255.
        mask = x['segmentation_mask'] - 1
        mask = self.to_one_hot(mask) if self.one_hot else mask
        label = x['species']
        id_image =  x['file_name']
        return img, mask, label, id_image 

    def __call__(self,):
        return self.train, self.val, self.test 