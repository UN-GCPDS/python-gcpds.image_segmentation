"""
=========
Zea Mays Seeds
=========
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


class ZeaMaysSeeds:
    already_unzipped = False
    def __init__(self, split=[0.2,0.2], seed: int=42, invert_back_ground_class=True,
                        id_: str='1POAEByzz8cNaiR0qHNwJo2Z6QD-lFUyD'):
        self.split = listify(split)
        self.seed = seed 
        self.invert_back_ground_class = invert_back_ground_class

        self.__id = id_
        self.__folder = os.path.join(os.path.dirname(__file__),
                                     'Datasets','ZeaMaysSeeds')
        self.__path_images =  os.path.join(self.__folder,
                                            'ZeaMays','JPEGImages')

        self.__path_masks =  os.path.join(self.__folder,
                                            'ZeaMays','SegmentationClass')

        self.__path_labelmap =  os.path.join(self.__folder,
                                            'ZeaMays','labelmap.txt')

        if not ZeaMaysSeeds.already_unzipped:
            self.__set_env()
            ZeaMaysSeeds.already_unzipped = True

        self.file_images = glob(os.path.join(self.__path_images, '*'))
        self.file_images = list(map(lambda x: x[:-4], self.file_images))
        self.file_images = list(map(lambda x: os.path.basename(x), self.file_images))
        self.file_images.sort()
        self.num_samples = len(self.file_images)

    def __set_env(self):
        destination_path_zip = os.path.join(self.__folder,
                                            'ZeaMaysSeeds.zip')
        os.makedirs(self.__folder, exist_ok=True)
        download_from_drive(self.__id, destination_path_zip)
        unzip(destination_path_zip, self.__folder)

    def __preprocessing_mask(self, mask):
        with open(self.__path_labelmap, "r") as FILE:
            lines = FILE.readlines()

        labels = {x.split(":")[0]: x.split(":")[1] for x in lines[1:]}

        for key in labels:
            labels[key] = np.array(labels[key].split(",")).astype("uint8")
        assert type(labels) == dict, "labels variable should be a dictionary"

        maskRGB = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
        if maskRGB.dtype == "float32":
            maskRGB = tf.cast(maskRGB*255, dtype="uint8")

        maskCategorical = tf.zeros(maskRGB.shape[0:2] , dtype="float32")
        for i, key in enumerate(labels):
            maskCategorical = tf.where(np.all(maskRGB == labels[key], axis=-1), i, maskCategorical)
        maskCategorical = tf.cast(maskCategorical, dtype="uint8")

        mask = tf.one_hot(maskCategorical, depth=3)

        if self.invert_back_ground_class == True:
            back_ground = mask[...,2] == 0
            back_ground = tf.cast(back_ground, tf.float32)
            back_ground = tf.expand_dims(back_ground, axis=-1)
            mask = tf.concat([mask[...,0:2],back_ground], axis=-1) #Ch 1: No germinate, Ch 2: germinate, Ch 3: Background

        mask = tf.cast(mask, tf.float32)
        return mask

    def load_instance_by_id(self, id_img):
        return self.load_instance(id_img)

    def load_instance(self, id_img):
        path_img = os.path.join(self.__path_images,id_img)
        path_mask = os.path.join(self.__path_masks,id_img)
        img = cv2.imread(f'{path_img}.jpg')/255
        mask = cv2.imread(f'{path_mask}.png')
        mask = self.__preprocessing_mask(mask, )
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