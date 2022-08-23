import os
from glob import glob

import cv2
import numpy as np
import tensorflow as tf
from gcpds.image_segmentation.datasets.utils import download_from_drive, unzip


class NerveUtp:
    def __init__(self, split=None):
        self.__id = "1GewZspflKFgN7Clut5Xqr3E3CQLSfoYU"
        self.__folder = os.path.join(os.path.dirname(__file__),
                                     'Datasets','nerviosUTP')
        self.__path_images =  os.path.join(self.__folder,
                                            'ImagenesNervios_')

        destination_path_zip = os.path.join(self.__folder,'ImagenesNervios.zip')
        os.makedirs(self.__folder,exist_ok=True)
        download_from_drive(self.__id, destination_path_zip)
        unzip(destination_path_zip, self.__folder)

        self.file_images = glob(os.path.join(self.__path_images,'*[!(mask)].png'))
        self.file_images = list(map(lambda x: x[:-4],self.file_images))
        self.split = split

    @staticmethod
    def __preprocessing_mask(mask):
        mask = mask[...,0] > 0.5 
        mask = mask.astype(np.float32)
        return mask[...,None]

    @staticmethod
    def __gen_dataset(file_images):
        def generator():
            for root_name in file_images:
                img = cv2.imread(f'{root_name}.png')/255
                mask = cv2.imread(f'{root_name}_mask.png')
                mask = NerveUtp.__preprocessing_mask(mask)
                label = os.path.split(root_name)[-1].split('_')[0]
                yield img, mask, label
        return generator

    
    def __generate_tf_data(self,files):
        return tf.data.Dataset.from_generator(self.__gen_dataset(files),
                                    output_signature = (tf.TensorSpec((None,None,None), tf.float32), 
                                                        tf.TensorSpec((None,None,None), tf.float32),
                                                        tf.TensorSpec(None, tf.string)))

    def __call__(self,):
        if self.split: 
            index = int(len(self.file_images)*self.split)
            train_files = self.file_images[:index]
            test_files = self.file_images[index:]
            train_dataset = self.__generate_tf_data(train_files)
            test_dataset = self.__generate_tf_data(test_files)
            return train_dataset, test_dataset
        else:
            return self.__generate_tf_data(self.file_images)









