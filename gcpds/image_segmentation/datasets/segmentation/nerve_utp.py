import os
from glob import glob

import tensorflow as tf
from gcpds.image_segmentation.datasets.utils import download_from_drive, unzip
from matplotlib import image


class NerveUtp:
    def __init__(self, split=0.2):
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
        self.file_images = map(lambda x: x[:-4],file_images)

    def gen_dataset(self,):
        def generator():
            for root_name in self.file_images:
                img = image.imread(f'{root_name}.png')
                mask = image.imread(f'{root_name}_mask.png')[...,None]
                label = os.path.split(root_name)[-1].split('_')[0]
                yield img, mask, label
        return generator


    def __call__(self,):
        return tf.data.Dataset.from_generator(self.gen_dataset(),
                                    output_signature = (tf.TensorSpec((None,None,None), tf.float32), 
                                                        tf.TensorSpec((None,None,None), tf.float32),
                                                        tf.TensorSpec(None, tf.string)))









