"""
Nerve-UTP: This dataset was acquired by the Universidad Tecnológica de
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
femoral nerve images (1857 input samples).


Random Fourier Features-Based Deep Learning Improvement with Class 
Activation Interpretability for Nerve Structure Segmentation
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8617795/
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



class NerveUtp:
    def __init__(self, split=1.0, seed=42):
        self.split = listify(split)
        self.seed = seed 

        self.__id = "1GewZspflKFgN7Clut5Xqr3E3CQLSfoYU"
        self.__folder = os.path.join(os.path.dirname(__file__),
                                     'Datasets','nerviosUTP')
        self.__path_images =  os.path.join(self.__folder,
                                            'ImagenesNervios_')
        self.__set_env()

        self.file_images = glob(os.path.join(self.__path_images, '*[!(mask)].png'))
        self.file_images = list(map(lambda x: x[:-4], self.file_images))
        self.file_images.sort()
        np.random.seed(seed)  
        np.random.shuffle(self.file_images)

        self.num_samples = len(self.file_images)
        self.labels_info = self.__get_labels_info()

    def __set_env(self):
        destination_path_zip = os.path.join(self.__folder,
                                            'ImagenesNervios.zip')
        os.makedirs(self.__folder, exist_ok=True)
        download_from_drive(self.__id, destination_path_zip)
        unzip(destination_path_zip, self.__folder)

    def __get_labels_info(self,):
        unique_labels = map(lambda x: os.path.split(x)[-1].split('_')[0],
                            self.file_images)
        unique_labels, counts = np.unique(list(unique_labels), return_counts=True)
        labels_info = {label:count for label,count in zip(unique_labels,counts)}
        return labels_info

    @staticmethod
    def __preprocessing_mask(mask):
        mask = mask[...,0] > 0.5 
        mask = mask.astype(np.float32)
        return mask[...,None]

    def load_instance(self, id_img):
        root_name = os.path.join(self.__path_images, id_img)
        return self.__load_instance(root_name)

    @staticmethod
    def __load_instance(root_name):
        img = cv2.imread(f'{root_name}.png')/255
        mask = cv2.imread(f'{root_name}_mask.png')
        mask = NerveUtp.__preprocessing_mask(mask)
        id_image = os.path.split(root_name)[-1]
        label = id_image.split('_')[0]
        return img, mask, label, id_image 


    @staticmethod
    def __gen_dataset(file_images):
        def generator():
            for root_name in file_images:
                yield self.__load_instance(root_name)
        return generator

    def __generate_tf_data(self,files):
        output_signature = (tf.TensorSpec((None,None,None), tf.float32), 
                            tf.TensorSpec((None,None,None), tf.float32),
                            tf.TensorSpec(None, tf.string),
                            tf.TensorSpec(None, tf.string))

        return tf.data.Dataset.from_generator(self.__gen_dataset(files),
                                    output_signature = output_signature)


    def __get_indices_partition(self):
        indices = [self.num_samples*s for s in self.split]
        indices = np.cumsum(indices)
        indices = np.round(indices)
        return indices.astype(np.int)

    def __get_log_tf_data(self,i,files):
        print(f' Number of images for Partition {i}: {len(files)}')
        return self.__generate_tf_data(files) 

    def __call__(self,):
        indices = self.__get_indices_partition()
        p_files = np.split(self.file_images,indices)
        p_files = [p_file for p_file in p_files if p_file.size]

        partitions = [self.__get_log_tf_data(i+1,p) for i,p in enumerate(p_files)]

        if len(partitions) ==1:
            return partitions[0]
        
        return partitions