import os 
from tensorflow.keras.preprocessing import image_dataset_from_directory 

class CatsVsDogs:
    def __init__(self,image_size = (180, 180)):
        self.__url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip" 
        self.__dirname = 'data'
        self.__filename = 'data.zip'
        self.__path = os.path.join(self.__dirname,self.__filename)
        os.makedirs(self.__dirname,exist_ok=True)
        self.__download()
        self.__unzip()
        self.__remove_corrupted()

        self.image_size = image_size 


    def __download(self,):
        cmm = f'wget {self.__url} -O {self.__path}' 
        os.system(cmm)
    
    def __unzip(self,):
        cmm = f'unzip -q {self.__path} -d {self.__dirname}'
        os.system(cmm)

    def __remove_corrupted(self,):
        num_skipped = 0
        for folder_name in ("Cat", "Dog"):
            folder_path = os.path.join(self.__dirname,"PetImages", folder_name)
            for fname in os.listdir(folder_path):
                fpath = os.path.join(folder_path, fname)
                try:
                    fobj = open(fpath, "rb")
                    is_jfif = bytes("JFIF",'utf-8') in fobj.peek(10)
                finally:
                    fobj.close()
                if not is_jfif:
                    num_skipped += 1
                    os.remove(fpath)
        print("Deleted %d images" % num_skipped)


    def load_data(self,batch_size,validation_split=0.3):
        path = os.path.join(self.__dirname,"PetImages")
        train_ds = image_dataset_from_directory(path,
                                                validation_split=validation_split,
                                                subset="training",
                                                seed=1337,
                                                image_size=self.image_size,
                                                batch_size=batch_size,
                                                )
        test_ds = image_dataset_from_directory(path,
                                              validation_split=validation_split,
                                              subset="validation",
                                              seed=1337,
                                              image_size=self.image_size,
                                              batch_size=batch_size,
                                              )
        return train_ds,test_ds
