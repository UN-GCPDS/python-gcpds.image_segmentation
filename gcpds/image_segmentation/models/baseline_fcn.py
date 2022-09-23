"""
https://github.com/cralji/RFF-Nerve-UTP/blob/main/FCN_Nerve-UTP.ipynb
"""

from functools import partial

import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers


DefaultConv2D = partial(layers.Conv2D,
                        kernel_size=3, activation='relu', padding="same")

DefaultPooling = partial(layers.MaxPool2D,
                        pool_size=2)

DefaultTranspConv = partial(layers.Conv2DTranspose,
                            kernel_size=3, strides=2,
                            padding='same',
                            use_bias=False, activation='relu')


def kernel_initializer(seed):
    return tf.keras.initializers.GlorotUniform(seed=seed)


def fcn_baseline(input_shape=(128,128,3), name='FCN', out_channels=1, out_ActFunction='sigmoid'):
    # Encoder 
    input_ = layers.Input(shape=input_shape)

    x =  layers.BatchNormalization(name='Batch00')(input_)
    
    x =  DefaultConv2D(32,kernel_initializer=kernel_initializer(34),name='Conv10')(x)
    x =  DefaultConv2D(32,kernel_initializer=kernel_initializer(4),name='Conv11')(x)
    x =  layers.BatchNormalization(name='Batch10')(x)
    x = DefaultPooling(name='Pool10')(x) # 128x128 -> 64x64

    x =  DefaultConv2D(32,kernel_initializer=kernel_initializer(56),name='Conv20')(x)
    x =  DefaultConv2D(32,kernel_initializer=kernel_initializer(28),name='Conv21')(x)
    x =  layers.BatchNormalization(name='Batch20')(x)
    x = DefaultPooling(name='Pool20')(x) # 64x64 -> 32x32

    x =  DefaultConv2D(64,kernel_initializer=kernel_initializer(332),name='Conv30')(x)
    x =  DefaultConv2D(64,kernel_initializer=kernel_initializer(2),name='Conv31')(x)
    x =  layers.BatchNormalization(name='Batch30')(x)
    x = level_1 = DefaultPooling(name='Pool30')(x) # 32x32 -> 16x16

    x =  DefaultConv2D(128,kernel_initializer=kernel_initializer(67),name='Conv40')(x)
    x =  DefaultConv2D(128,kernel_initializer=kernel_initializer(89),name='Conv41')(x)
    x =  layers.BatchNormalization(name='Batch40')(x)
    x = level_2 = DefaultPooling(name='Pool40')(x) # 16x16 -> 8x8

    x =  DefaultConv2D(256,kernel_initializer=kernel_initializer(7),name='Conv50')(x)
    x =  DefaultConv2D(256,kernel_initializer=kernel_initializer(23),name='Conv51')(x)
    x =  layers.BatchNormalization(name='Batch50')(x)
    x =  DefaultPooling(name='Pool50')(x) # 8x8 -> 4x4

    
    #Decoder
    x = level_3 = DefaultTranspConv(out_channels,kernel_size=4,
                                    use_bias=False, 
                                    kernel_initializer=kernel_initializer(98),
                                    name='Trans60')(x)
    x = DefaultConv2D(out_channels,kernel_size=1,
                    activation=None,kernel_initializer=kernel_initializer(75),
                    name='Conv60')(level_2)


    x =  layers.Add(name='Add10')([x,level_3])

    
    x = level_4 = DefaultTranspConv(out_channels,kernel_size=4,use_bias=False,
                                    kernel_initializer=kernel_initializer(87),
                                    name='Trans70')(x)
    x = DefaultConv2D(out_channels,kernel_size=1,activation=None,
                        kernel_initializer=kernel_initializer(54),
                        name='Conv70')(level_1)

    x =  layers.Add(name='Add20')([x,level_4])

    x = DefaultTranspConv(out_channels,kernel_size=16,strides=8,
                            activation=out_ActFunction,use_bias=True,
                            kernel_initializer=kernel_initializer(32),
                            name='Trans80')(x)


    model = Model(input_,x,name=name)

    return model 



if __name__ == "__main__":
    model = fcn_baseline()
    model.summary()
