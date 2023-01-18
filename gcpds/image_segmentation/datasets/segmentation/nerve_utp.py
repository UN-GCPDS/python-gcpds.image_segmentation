"""
=========
Nerve-UTP
=========

This dataset was acquired by the Universidad Tecnológica de
Pereira (https://www.utp.edu.co, accessed on 17 November 2021) and the
Santa Mónica Hospital, Dosquebradas, Colombia. It contains 691 images
of the following nerve structures: the sciatic nerve (287 instances),
the ulnar nerve (221 instances), the median nerve (41 instances), and 
the femoral nerve (70 instances). A SONOSITE Nano-Maxx device was used, 
fixing a 640 × 480 pixel resolution. Each image was labeled by an 
anesthesiologist from the Santa Mónica Hospital. As prepossessing, 
morphological operations such as dilation and erosion were applied. 
Next, we defined a region of interest by computing the bounding box 
around each nerve structure. As a result, we obtained images holding 
a maximum resolution of 360 × 279 pixels. Lastly, we applied a data 
augmentation scheme to obtain the following samples: 861 sciatic nerve
images, 663 ulnar nerve images, 123 median nerve images, and 210 
femoral nerve images (1857 input samples)[1].


.. [1] `Random Fourier Features-Based Deep Learning Improvement with Class Activation Interpretability for Nerve Structure Segmentation`_

.. _`Random Fourier Features-Based Deep Learning Improvement with Class Activation Interpretability for Nerve Structure Segmentation`: http://www.sdss.org/dr14/help/glossary/#stripe
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
from collections import Counter


class NerveUtp:
    already_unzipped = False
    def __init__(self, split=[0.2,0.2], seed: int=42):
        self.split = listify(split)
        self.seed = seed 

        self.__id = "1GewZspflKFgN7Clut5Xqr3E3CQLSfoYU"
        self.__folder = os.path.join(os.path.dirname(__file__),
                                     'Datasets','nerviosUTP')
        self.__path_images =  os.path.join(self.__folder,
                                            'ImagenesNervios_')
                                            
        if not NerveUtp.already_unzipped:
            self.__set_env()
            NerveUtp.already_unzipped = True

        self.file_images = glob(os.path.join(self.__path_images, '*[!(mask)].png'))
        self.file_images = list(map(lambda x: x[:-4], self.file_images))
        self.file_images.sort()

        self.labels =  list(map(lambda x: os.path.split(x)[-1].split('_')[0],
                            self.file_images))

    
        self.num_samples = len(self.file_images)
        self.labels_info = Counter(self.labels)

    def __set_env(self):
        destination_path_zip = os.path.join(self.__folder,
                                            'ImagenesNervios.zip')
        os.makedirs(self.__folder, exist_ok=True)
        download_from_drive(self.__id, destination_path_zip)
        unzip(destination_path_zip, self.__folder)

    @staticmethod
    def __preprocessing_mask(mask):
        mask = mask[...,0] > 0.5 
        mask = mask.astype(np.float32)
        return mask[...,None]

    def load_instance_by_id(self, id_img):
        root_name = os.path.join(self.__path_images, id_img)
        return self.load_instance(root_name)

    @staticmethod
    def load_instance(root_name):
        img = cv2.imread(f'{root_name}.png')/255
        img = img[...,0][...,None]
        mask = cv2.imread(f'{root_name}_mask.png')
        mask = NerveUtp.__preprocessing_mask(mask)
        id_image = os.path.split(root_name)[-1]
        label = id_image.split('_')[0]
        return img, mask, label, id_image 


    @staticmethod
    def __gen_dataset(file_images):
        def generator():
            for root_name in file_images:
                yield NerveUtp.load_instance(root_name)
        return generator

    def __generate_tf_data(self,files):
        output_signature = (tf.TensorSpec((None,None,None), tf.float32), 
                            tf.TensorSpec((None,None,None), tf.float32),
                            tf.TensorSpec(None, tf.string),
                            tf.TensorSpec(None, tf.string))

        dataset = tf.data.Dataset.from_generator(self.__gen_dataset(files),
                                    output_signature = output_signature)

        len_files = len(files)
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(len_files))
        return dataset


    def __get_log_tf_data(self,i,files):
        print(f' Number of images for Partition {i}: {len(files)}')
        return self.__generate_tf_data(files) 

    def __call__(self,):

        train_imgs, test_imgs, l_train, _ = train_test_split(self.file_images,
                                                            self.labels,
                                                            test_size=self.split[0],
                                                            stratify = self.labels,
                                                            random_state=self.seed)

        train_imgs, val_imgs, _ , _ = train_test_split(train_imgs, l_train,
                                                           test_size=self.split[1],
                                                           stratify = l_train,
                                                           random_state=self.seed )

        
        p_files = [train_imgs, val_imgs, test_imgs]
        
        partitions = [self.__get_log_tf_data(i+1,p) for i,p in enumerate(p_files)]

        return partitions