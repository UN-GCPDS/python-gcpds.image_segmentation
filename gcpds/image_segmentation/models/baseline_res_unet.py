"""
https://github.com/cralji/RFF-Nerve-UTP/blob/main/Res-UNET_Nerve-UTP.ipynb
"""

from tensorflow.keras import Model, layers, initializers


def kernel_initializer(seed):
    return initializers.GlorotUniform(seed=seed)


def upsample_conv(filters, kernel_size, strides, padding, kernel_initializer, name):
    return layers.Conv2DTranspose(filters, kernel_size,
                                 strides=strides,
                                 kernel_initializer=kernel_initializer,
                                 padding=padding,
                                 name=name)


def res_block(x,units,kernel_initializer,name):
    x_c = x
    x = layers.Conv2D(units,(1,1),(1,1),
                      kernel_initializer=kernel_initializer,
                      padding='same',name=f'{name}_Conv00')(x)
    x = layers.BatchNormalization(name=f'{name}_Batch00')(x)
    x = layers.Activation('relu',name=f'{name}_Act00')(x)
    x = layers.Conv2D(units,(3,3),(1,1),
                      kernel_initializer=kernel_initializer,
                      padding='same',
                      name=f'{name}_Conv01')(x)
    x = layers.BatchNormalization(name=f'{name}_Batch01')(x)
    x_c = layers.Conv2D(units,(1,1),(1,1),
                        kernel_initializer=kernel_initializer,
                        padding='same',
                        name=f'{name}_Conv02')(x_c)
    x_c = layers.BatchNormalization(name=f'{name}_Batch02')(x_c)
    x = layers.Add(name=f'{name}_Add00')([x,x_c])
    x = layers.Activation('relu',name=f'{name}_Act01')(x)
    return x


def res_unet_baseline(input_shape=(128,128,3), name='RES_UNET', out_channels=1, out_ActFunction='sigmoid'):
    input_ = layers.Input(shape=input_shape, name='input')

    pp_in_layer = input_

    pp_in_layer = layers.BatchNormalization()(pp_in_layer)
    c1 = res_block(pp_in_layer,8,kernel_initializer=kernel_initializer(34),name='Res00')
    c1 = res_block(c1,8,kernel_initializer=kernel_initializer(3),name='Res01')
    p1 = layers.MaxPooling2D((2, 2),name='Maxp00') (c1)

    c2 = res_block(p1,16,kernel_initializer=kernel_initializer(7),name='Res02')
    c2 = res_block(c2,16,kernel_initializer=kernel_initializer(98),name='Res03')
    p2 = layers.MaxPooling2D((2, 2),name='Maxp01') (c2)

    c3 = res_block(p2,32,kernel_initializer=kernel_initializer(5),name='Res04')
    c3 = res_block(c3,32,kernel_initializer=kernel_initializer(23),name='Res05')
    p3 = layers.MaxPooling2D((2, 2),name='Maxp02') (c3)

    c4 = res_block(p3,64,kernel_initializer=kernel_initializer(32),name='Res06')
    c4 = res_block(c4,64,kernel_initializer=kernel_initializer(43),name='Res07')
    p4 = layers.MaxPooling2D(pool_size=(2, 2),name='Maxp03') (c4)
    # Bottle Neck
    c5 = res_block(p4,128,kernel_initializer=kernel_initializer(43),name='Res08')
    c5 = res_block(c5,128,kernel_initializer=kernel_initializer(65),name='Res09')
    # upsampling
    u6 = upsample_conv(64, (2, 2), strides=(2, 2),
                       padding='same',
                       kernel_initializer=kernel_initializer(4),
                       name='Upsam00') (c5)
    u6 = layers.concatenate([u6, c4],name='Concat00')
    c6 = res_block(u6,64,kernel_initializer=kernel_initializer(65),name='Res10')
    c6 = res_block(c6,64,kernel_initializer=kernel_initializer(87),name='Res11')

    u7 = upsample_conv(32, (2, 2), strides=(2, 2), 
                       padding='same',
                       kernel_initializer=kernel_initializer(2),
                       name='Upsam01') (c6)
    u7 = layers.concatenate([u7, c3],name='Concat01')
    c7 = res_block(u7,32,kernel_initializer=kernel_initializer(34),name='Res12')
    c7 = res_block(c7,32,kernel_initializer=kernel_initializer(4),name='Res13')

    u8 = upsample_conv(16, (2, 2), strides=(2, 2),
                       padding='same',
                       kernel_initializer=kernel_initializer(432),
                       name='Upsam02') (c7)
    u8 = layers.concatenate([u8, c2],name='Concat02')
    c8 = res_block(u8,16,kernel_initializer=kernel_initializer(32),name='Res14')
    c8 = res_block(c8,16,kernel_initializer=kernel_initializer(42),name='Res15')

    u9 = upsample_conv(8, (2, 2), strides=(2, 2), 
                       padding='same',
                       kernel_initializer=kernel_initializer(32),
                       name='Upsam03') (c8)
    u9 = layers.concatenate([u9, c1], axis=3,name='Concat03')
    c9 = res_block(u9,8,kernel_initializer=kernel_initializer(4),name='Res16')
    c9 = res_block(c9,8,kernel_initializer=kernel_initializer(6),name='Res17')

    d = layers.Conv2D(out_channels, kernel_size=(1, 1), activation=out_ActFunction,name='Output') (c9)
    
    seg_model = Model(inputs=[input_], outputs=[d])
    
    return seg_model

if __name__ == '__main__':
    model = res_unet_baseline()
    model.summary()