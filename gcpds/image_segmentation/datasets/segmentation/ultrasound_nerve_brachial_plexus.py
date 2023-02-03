"""
=====================
Nerve segment dataset
=====================

`This dataset belongs to the Kaggle Competition repository`_. It holds labeled ultrasound images of 
the neck concerning the brachial plexus (BP). In particular,
47 different subjects were studied, recording 119 to 580 images 
per subject (5635 as a whole) at 420 × 580 pixel resolution. 
For concrete testing, we performed a pruning procedure to remove
images with inconsistent annotations, yielding to 2323 samples[1].


.. [1] `Random Fourier Features-Based Deep Learning Improvement with Class Activation Interpretability for Nerve Structure Segmentation`_

.. _`Random Fourier Features-Based Deep Learning Improvement with Class Activation Interpretability for Nerve Structure Segmentation`: http://www.sdss.org/dr14/help/glossary/#stripe

.. _`This dataset belongs to the Kaggle Competition repository`: https://www.kaggle.com/competitions/ultrasound-nerve-segmentation/data
"""


import os
from glob import glob

import cv2
import numpy as np
import tensorflow as tf
from gcpds.image_segmentation.datasets.utils import download_from_drive
from gcpds.image_segmentation.datasets.utils import unzip
from gcpds.image_segmentation.datasets.utils import listify
from sklearn.model_selection import train_test_split



class BrachialPlexus:
    already_unzipped = False 
    def __init__(self, split=[0.2,0.2], seed: int=42,
                        id_: str='1In6sQgN2on6PMdqS34K88fvHJwBeak3I'):

        self.split = listify(split)
        self.seed = seed 

        self.__id = id_
        self.__folder = os.path.join(os.path.dirname(__file__),
                                     'Datasets','BrachialPlexus')
        self.__path_images =  self.__folder

        if not BrachialPlexus.already_unzipped:
            self.__set_env()
            BrachialPlexus.already_unzipped = True

        self.file_images = glob(os.path.join(self.__path_images, '*[!(mask)].*'))
        self.file_images = map(lambda x: x[:-4], self.file_images)
        self.file_images = list(filter(lambda x: self._filter_mask(x), self.file_images))
        self.file_images.sort()

        self.num_samples = len(self.file_images)

    def _filter_mask(self,file_path):
        mask = cv2.imread(f'{file_path}_mask.tif')
        uniques = np.unique(mask)
        return len(uniques) == 2 

    def __set_env(self):
        destination_path_zip = os.path.join(self.__folder,
                                            'BrachialPlexus.zip')
        os.makedirs(self.__folder, exist_ok=True)
        download_from_drive(self.__id, destination_path_zip)
        unzip(destination_path_zip, self.__folder)

    @staticmethod
    def __preprocessing_mask(mask):
        mask = mask[...,0]/255
        return mask[...,None]

    def load_instance_by_id(self, id_img):
        root_name = os.path.join(self.__path_images, id_img)
        return self.load_instance(root_name)

    @staticmethod
    def load_instance(root_name):
        img = cv2.imread(f'{root_name}.tif')/255
        img = img[...,0][...,None]
        mask = cv2.imread(f'{root_name}_mask.tif')
        mask = BrachialPlexus.__preprocessing_mask(mask)
        id_image = os.path.split(root_name)[-1]
        return img, mask, id_image 

    @staticmethod
    def __gen_dataset(file_images):
        def generator():
            for root_name in file_images:
                yield BrachialPlexus.load_instance(root_name)
        return generator

    def __generate_tf_data(self,files):
        output_signature = (tf.TensorSpec((None,None,None), tf.float32), 
                            tf.TensorSpec((None,None,None), tf.float32),
                            tf.TensorSpec(None, tf.string),
                            )

        dataset = tf.data.Dataset.from_generator(self.__gen_dataset(files),
                                    output_signature = output_signature)

        len_files = len(files)
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(len_files))
        return dataset

    def __get_log_tf_data(self,i,files):
        print(f' Number of images for Partition {i}: {len(files)}')
        return self.__generate_tf_data(files) 

    def __call__(self,):
        train_imgs, test_imgs = train_test_split(self.file_images,
                                                test_size=self.split[0],
                                                random_state=self.seed)

        train_imgs, val_imgs  = train_test_split(train_imgs,
                                                 test_size=self.split[1],
                                                 random_state=self.seed )

        
        p_files = [train_imgs, val_imgs, test_imgs]
        
        partitions = [self.__get_log_tf_data(i+1,p) for i,p in enumerate(p_files)]

        return partitions