import tensorflow as tf 
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def upsample(filters,size,strides=2,padding="same",batchnorm=False,dropout=0):

    layer = tf.keras.Sequential()
    layer.add(
        tf.keras.layers.Conv2DTranspose(filters,size,strides,padding,use_bias = False))

    if batchnorm:
        layer.add(tf.keras.layers.BatchNormalization())

    if dropout != 0:
        layer.add(tf.keras.layers.Dropout(dropout))

    layer.add(tf.keras.layers.ReLU())

    return layer
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def get_encoder(input_shape=[None,None,3], trainable = True, name="encoder"): 
    Input = tf.keras.layers.Input(shape=input_shape)
    base_model = tf.keras.applications.MobileNetV2(input_tensor=Input, include_top=False)
    layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    encoder  = tf.keras.Model(inputs=Input, outputs=layers,name=name)
    encoder.trainable = trainable

    return encoder 
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def get_decoder(skips,dropout=0):
    up_stack = [
        upsample(512, 3,dropout=dropout),  # 4x4 -> 8x8
        upsample(256, 3,dropout=dropout),  # 8x8 -> 16x16
        upsample(128, 3,dropout=dropout),  # 16x16 -> 32x32
        upsample(64, 3,dropout=dropout),   # 32x32 -> 64x64
    ]
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up,skip in zip(up_stack,skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x,skip])
    return x
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def unet_mobilenet(input_shape=(128,128,3), out_channels=1, out_ActFunction='sigmoid', trainable = False, name="unetMobile"):
    input = tf.keras.layers.Input(shape=input_shape)

    skips = get_encoder(input_shape=list(input.shape[1:]),  trainable = trainable)(input)

    x = get_decoder(skips, dropout=0)

    last = tf.keras.layers.Conv2DTranspose(
        out_channels, kernel_size=(1,1), strides=2,
        padding='same',activation=out_ActFunction)  #64x64 -> 128x128

    x = last(x)
    model = tf.keras.Model(inputs=input, outputs=x,name=name)
    return model
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    model = unet_mobilenet()
    model.summary()