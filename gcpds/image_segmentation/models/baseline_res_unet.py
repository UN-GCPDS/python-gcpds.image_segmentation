
from tensorflow.keras import Model, layers


def upsample_conv(filters, kernel_size, strides, padding):
    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)


def res_block(x,units):
    x_c = x
    x = layers.Conv2D(units,(1,1),(1,1),padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(units,(3,3),(1,1),padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x_c = layers.Conv2D(units,(1,1),(1,1),padding='same')(x_c)
    x_c = layers.BatchNormalization()(x_c)
    x = layers.Add()([x,x_c])
    x = layers.Activation('relu')(x)
    return x


def res_unet_baseline(input_shape=(128,128,3), name='RES_UNET'):
    input_ = layers.Input(shape=input_shape)

    pp_in_layer = input_

    pp_in_layer = layers.BatchNormalization()(pp_in_layer)
    c1 = res_block(pp_in_layer,8)
    c1 = res_block(c1,8)
    p1 = layers.MaxPooling2D((2, 2)) (c1)

    c2 = res_block(p1,16)
    c2 = res_block(c2,16)
    p2 = layers.MaxPooling2D((2, 2)) (c2)

    c3 = res_block(p2,32)
    c3 = res_block(c3,32)
    p3 = layers.MaxPooling2D((2, 2)) (c3)

    c4 = res_block(p3,64)
    c4 = res_block(c4,64)
    p4 = layers.MaxPooling2D(pool_size=(2, 2)) (c4)
    # Bottle Neck
    c5 = res_block(p4,128)
    c5 = res_block(c5,128)
    # upsampling
    u6 = upsample_conv(64, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = layers.concatenate([u6, c4])
    c6 = res_block(u6,64)
    c6 = res_block(c6,64)

    u7 = upsample_conv(32, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = layers.concatenate([u7, c3])
    c7 = res_block(u7,32)
    c7 = res_block(c7,32)

    u8 = upsample_conv(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = layers.concatenate([u8, c2])
    c8 = res_block(u8,16)
    c8 = res_block(c8,16)

    u9 = upsample_conv(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = res_block(u9,8)
    c9 = res_block(c9,8)

    d = layers.Conv2D(1, (1, 1), activation='sigmoid') (c9)
    
    seg_model = Model(inputs=[input_], outputs=[d])
    
    return seg_model

if __name__ == '__main__':
    model = res_unet_baseline()
    model.summary()