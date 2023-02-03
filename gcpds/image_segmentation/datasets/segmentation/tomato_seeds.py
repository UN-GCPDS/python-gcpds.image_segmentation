"""
=========
Tomato Seeds
=========
"""


import logging
import os
from glob import glob

import cv2
import numpy as np
import tensorflow as tf
from gcpds.image_segmentation.datasets.utils import download_from_drive
from gcpds.image_segmentation.datasets.utils import unzip
from gcpds.image_segmentation.datasets.utils import listify
from sklearn.model_selection import train_test_split


class TomatoSeeds:
    already_unzipped = False
    def __init__(self, split=[0.2,0.2], seed: int=42,
                        id_: str='1J-jjASPC0VtibEj1_2MJ_lnhuP_ltvgY'):
        self.split = listify(split)
        self.seed = seed 

        self.__id = id_
        self.__folder = os.path.join(os.path.dirname(__file__),
                                     'Datasets','tomatoSeeds')
        self.__path_images =  os.path.join(self.__folder,
                                            'DatasetE2','JPEGImages')

        self.__path_masks =  os.path.join(self.__folder,
                                            'DatasetE2','SegmentationClass')

        if not TomatoSeeds.already_unzipped:
            self.__set_env()
            TomatoSeeds.already_unzipped = True

        self.file_images = glob(os.path.join(self.__path_images, '*'))
        self.file_images = list(map(lambda x: x[:-4], self.file_images))
        self.file_images = list(map(lambda x: os.path.basename(x), self.file_images))
        self.file_images.sort()
        self.num_samples = len(self.file_images)

    def __set_env(self):
        destination_path_zip = os.path.join(self.__folder,
                                            'TomatoSeeds.zip')
        os.makedirs(self.__folder, exist_ok=True)
        download_from_drive(self.__id, destination_path_zip)
        unzip(destination_path_zip, self.__folder)

    @staticmethod
    def __preprocessing_mask(mask):
        mask = mask == 255 
        seed = mask[...,2] #bgr
        no_germinate = mask[...,2] & mask[...,1]
        getminate = mask[...,2] & ~mask[...,1]

        mask = np.concatenate([seed[...,None],
                               no_germinate[...,None],
                               getminate[...,None]],
                               axis=-1)
        mask = mask.astype(np.float32)
        return mask #BGR, B=Seed, G=No Germinate, R=germinate

    def load_instance_by_id(self, id_img):
        return self.load_instance(id_img)

    def load_instance(self, id_img):
        path_img = os.path.join(self.__path_images,id_img)
        path_mask = os.path.join(self.__path_masks,id_img)
        img = cv2.imread(f'{path_img}.jpg')/255
        mask = cv2.imread(f'{path_mask}.png')
        mask = self.__preprocessing_mask(mask)
        id_image = id_img
        return img, mask, id_image 

    def __gen_dataset(self, file_images):
        def generator():
            for root_name in file_images:
                yield self.load_instance(root_name)
        return generator

    def __generate_tf_data(self,files):
        output_signature = (tf.TensorSpec((None,None,None), tf.float32), 
                            tf.TensorSpec((None,None,None), tf.float32),
                            tf.TensorSpec(None, tf.string)
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

        train_imgs, test_imgs, *_ = train_test_split(self.file_images,
                                                            test_size=self.split[0],
                                                            random_state=self.seed)

        train_imgs, val_imgs, *_ = train_test_split(train_imgs,
                                                           test_size=self.split[1],
                                                           random_state=self.seed )

        
        p_files = [train_imgs, val_imgs, test_imgs]
        
        partitions = [self.__get_log_tf_data(i+1,p) for i,p in enumerate(p_files)]

        return partitions