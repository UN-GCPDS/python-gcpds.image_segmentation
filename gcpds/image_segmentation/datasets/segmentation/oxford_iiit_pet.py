"""
I only made this code to keep the interface (NervetUPT, ultrasound_nerve_brachial_plexus)
 (__call__, load_instance_by_id, and order of tuple), 
I still recommend using tfds to create the datasets
"""
import tensorflow as tf
import tensorflow_datasets as tfds

class OxfordIiitPet:
    def __init__(self, split=[70, 15, 15], seed: int=42):

        self.split=OxfordIiitPet._get_splits(split)
        dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True, split=self.split)
        self.info = info 
        self.train, self.val, self.test  = dataset

        self.train = self.train.map(lambda x: OxfordIiitPet._keep_interface(x)) 
        self.val= self.val.map(lambda x: OxfordIiitPet._keep_interface(x)) 
        self.test = self.test.map(lambda x: OxfordIiitPet._keep_interface(x)) 
        self.labels_info = {0:'cat', 1:'dog'}

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

    @staticmethod
    def _keep_interface(x):
        img = tf.cast(x['image'], tf.float32)/255.
        mask = x['segmentation_mask']
        label = x['species']
        id_image =  x['file_name']
        return img, mask, label, id_image 

    def __call__(self,):
        return self.train, self.val, self.test 