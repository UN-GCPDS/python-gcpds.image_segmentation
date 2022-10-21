import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from keras import backend as K
from keras.layers import Layer

class MaxPoolingWithArgmax2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding="same", **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if K.backend() == "tensorflow":
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = K.tf.nn.max_pool_with_argmax(
                inputs, ksize=ksize, strides=strides, padding=padding
            )
        else:
            errmsg = "{} backend is not supported for layer {}".format(
                K.backend(), type(self).__name__
            )
            raise NotImplementedError(errmsg)
        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim // ratio[idx] if dim is not None else None
            for idx, dim in enumerate(input_shape)
        ]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size
    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        with tf.compat.v1.variable_scope(self.name):
            mask = K.cast(mask, "int32")
            input_shape = K.tf.shape(updates, out_type="int32")
            #  calculation new shape
            if output_shape is None:
                output_shape = (
                    input_shape[0],
                    input_shape[1] * self.size[0],
                    input_shape[2] * self.size[1],
                    input_shape[3],
                )
            self.output_shape1 = output_shape

            # calculation indices for batch, height, width and feature maps
            one_like_mask = K.ones_like(mask, dtype="int32")
            batch_shape = K.concatenate([[input_shape[0]], [1], [1], [1]], axis=0)
            batch_range = K.reshape(
                K.tf.range(output_shape[0], dtype="int32"), shape=batch_shape
            )
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2]
            feature_range = K.tf.range(output_shape[3], dtype="int32")
            f = one_like_mask * feature_range

            # transpose indices & reshape update values to one dimension
            updates_size = K.tf.size(updates)
            indices = K.transpose(K.reshape(K.stack([b, y, x, f]), [4, updates_size]))
            values = K.reshape(updates, [updates_size])
            ret = K.tf.scatter_nd(indices, values, output_shape)
            return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
            mask_shape[0],
            mask_shape[1] * self.size[0],
            mask_shape[2] * self.size[1],
            mask_shape[3],
        )

def segnet_baseline(input_shape=(128,128,3), name='SEGNET', out_channels=1, out_ActFunction='sigmoid', kernel = 3, ActFunction = 'selu'):
    Input = tf.keras.Input(shape=input_shape, name='Input')
    #***********************************************************************Encoder***********************************************************************
    Conv1 = Conv2D(filters=64, kernel_size=(kernel,kernel), padding='same', activation=ActFunction, data_format='channels_last', name='Conv1')(Input)
    Norm1 = BatchNormalization(name='Norm1')(Conv1)
    Act1 = Activation(ActFunction, name=ActFunction+'1')(Norm1)
    Conv2 = Conv2D(filters=64, kernel_size=(kernel,kernel), padding='same', activation=ActFunction, data_format='channels_last', name='Conv2')(Act1)
    Norm2 = BatchNormalization(name='Norm2')(Conv2)
    Act2 = Activation(ActFunction, name=ActFunction+'2')(Norm2)
    Maxpool1, Argmax1 = MaxPoolingWithArgmax2D(name='Max2DArgmax1')(Act2)

    Conv3 = Conv2D(filters=128, kernel_size=(kernel,kernel), padding='same', activation=ActFunction, data_format='channels_last', name='Conv3')(Maxpool1)
    Norm3 = BatchNormalization(name='Norm3')(Conv3)
    Act3 = Activation(ActFunction, name=ActFunction+'3')(Norm3)
    Conv4 = Conv2D(filters=128, kernel_size=(kernel,kernel), padding='same', activation=ActFunction, data_format='channels_last', name='Conv4')(Act3)
    Norm4 = BatchNormalization(name='Norm4')(Conv4)
    Act4 = Activation(ActFunction, name=ActFunction+'4')(Norm4)
    Maxpool2, Argmax2 = MaxPoolingWithArgmax2D(name='Max2DArgmax2')(Act4)

    Conv5 = Conv2D(filters=256, kernel_size=(kernel,kernel), padding='same', activation=ActFunction, data_format='channels_last', name='Conv5')(Maxpool2)
    Norm5 = BatchNormalization(name='Norm5')(Conv5)
    Act5 = Activation(ActFunction, name=ActFunction+'5')(Norm5)
    Conv6 = Conv2D(filters=256, kernel_size=(kernel,kernel), padding='same', activation=ActFunction, data_format='channels_last', name='Conv6')(Act5)
    Norm6 = BatchNormalization(name='Norm6')(Conv6)
    Act6 = Activation(ActFunction, name=ActFunction+'6')(Norm6)
    Conv7 = Conv2D(filters=256, kernel_size=(kernel,kernel), padding='same', activation=ActFunction, data_format='channels_last', name='Conv7')(Act6)
    Norm7 = BatchNormalization(name='Norm7')(Conv7)
    Act7 = Activation(ActFunction, name=ActFunction+'7')(Norm7)
    Maxpool3, Argmax3 = MaxPoolingWithArgmax2D(name='Max2DArgmax3')(Act7)

    Conv8 = Conv2D(filters=512, kernel_size=(kernel,kernel), padding='same', activation=ActFunction, data_format='channels_last', name='Conv8')(Maxpool3)
    Norm8 = BatchNormalization(name='Norm8')(Conv8)
    Act8 = Activation(ActFunction, name=ActFunction+'8')(Norm8)
    Conv9 = Conv2D(filters=512, kernel_size=(kernel,kernel), padding='same', activation=ActFunction, data_format='channels_last', name='Conv9')(Act8)
    Norm9 = BatchNormalization(name='Norm9')(Conv9)
    Act9 = Activation(ActFunction, name=ActFunction+'9')(Norm9)
    Conv10 = Conv2D(filters=512, kernel_size=(kernel,kernel), padding='same', activation=ActFunction, data_format='channels_last', name='Conv10')(Act9)
    Norm10 = BatchNormalization(name='Norm10')(Conv10)
    Act10 = Activation(ActFunction, name=ActFunction+'10')(Norm10)
    Maxpool4, Argmax4 = MaxPoolingWithArgmax2D(name='Max2DArgmax4')(Act10)

    Conv11 = Conv2D(filters=512, kernel_size=(kernel,kernel), padding='same', activation=ActFunction, data_format='channels_last', name='Conv11')(Maxpool4)
    Norm11 = BatchNormalization(name='Norm11')(Conv11)
    Act11 = Activation(ActFunction, name=ActFunction+'11')(Norm11)
    Conv12 = Conv2D(filters=512, kernel_size=(kernel,kernel), padding='same', activation=ActFunction, data_format='channels_last', name='Conv12')(Act11)
    Norm12 = BatchNormalization(name='Norm12')(Conv12)
    Act12 = Activation(ActFunction, name=ActFunction+'12')(Norm12)
    Conv13 = Conv2D(filters=512, kernel_size=(kernel,kernel), padding='same', activation=ActFunction, data_format='channels_last', name='Conv13')(Act12)
    Norm13 = BatchNormalization(name='Norm13')(Conv13)
    Act13 = Activation(ActFunction, name=ActFunction+'13')(Norm13)
    Maxpool5, Argmax5 = MaxPoolingWithArgmax2D(name='Max2DArgmax5')(Act13)

    #******************************************************************Decoder*****************************************************************************************
    UnPool5 = MaxUnpooling2D(name='Unpool5')([Maxpool5, Argmax5], tf.shape(Conv13))
    Conv14 = Conv2D(filters=512, kernel_size=(kernel,kernel), padding='same', activation=ActFunction, data_format='channels_last', name='Conv14')(UnPool5)
    Norm14 = BatchNormalization(name='Norm14')(Conv14)
    Act14 = Activation(ActFunction, name=ActFunction+'14')(Norm14)
    Conv15 = Conv2D(filters=512, kernel_size=(kernel,kernel), padding='same', activation=ActFunction, data_format='channels_last', name='Conv15')(Act14)
    Norm15 = BatchNormalization(name='Norm15')(Conv15)
    Act15 = Activation(ActFunction, name=ActFunction+'15')(Norm15)
    Conv16 = Conv2D(filters=512, kernel_size=(kernel,kernel), padding='same', activation=ActFunction, data_format='channels_last', name='Conv16')(Act15)
    Norm16 = BatchNormalization(name='Norm16')(Conv16)
    Act16 = Activation(ActFunction, name=ActFunction+'16')(Norm16)

    UnPool4 = MaxUnpooling2D(name='Unpool4')([Act16, Argmax4], tf.shape(Conv10))
    Conv17 = Conv2D(filters=512, kernel_size=(kernel,kernel), padding='same', activation=ActFunction, data_format='channels_last', name='Conv17')(UnPool4)
    Norm17 = BatchNormalization(name='Norm17')(Conv17)
    Act17 = Activation(ActFunction, name=ActFunction+'17')(Norm17)
    Conv18 = Conv2D(filters=512, kernel_size=(kernel,kernel), padding='same', activation=ActFunction, data_format='channels_last', name='Conv18')(Act17)
    Norm18 = BatchNormalization(name='Norm18')(Conv18)
    Act18 = Activation(ActFunction, name=ActFunction+'18')(Norm18)
    Conv19 = Conv2D(filters=256, kernel_size=(kernel,kernel), padding='same', activation=ActFunction, data_format='channels_last', name='Conv19')(Act18)
    Norm19 = BatchNormalization(name='Norm19')(Conv19)
    Act19 = Activation(ActFunction, name=ActFunction+'19')(Norm19)

    UnPool3 = MaxUnpooling2D(name='Unpool3')([Act19, Argmax3], tf.shape(Conv7))
    Conv20 = Conv2D(filters=256, kernel_size=(kernel,kernel), padding='same', activation=ActFunction, data_format='channels_last', name='Conv20')(UnPool3)
    Norm20 = BatchNormalization(name='Norm20')(Conv20)
    Act20 = Activation(ActFunction, name=ActFunction+'20')(Norm20)
    Conv21 = Conv2D(filters=256, kernel_size=(kernel,kernel), padding='same', activation=ActFunction, data_format='channels_last', name='Conv21')(Act20)
    Norm21 = BatchNormalization(name='Norm21')(Conv21)
    Act21 = Activation(ActFunction, name=ActFunction+'21')(Norm21)
    Conv22 = Conv2D(filters=128, kernel_size=(kernel,kernel), padding='same', activation=ActFunction, data_format='channels_last', name='Conv22')(Act21)
    Norm22 = BatchNormalization(name='Norm22')(Conv22)
    Act22 = Activation(ActFunction, name=ActFunction+'22')(Norm22)

    UnPool2 = MaxUnpooling2D(name='Unpool2')([Act22, Argmax2], tf.shape(Conv4))
    Conv23 = Conv2D(filters=128, kernel_size=(kernel,kernel), padding='same', activation=ActFunction, data_format='channels_last', name='Conv23')(UnPool2)
    Norm23 = BatchNormalization(name='Norm23')(Conv23)
    Act23 = Activation(ActFunction, name=ActFunction+'23')(Norm23)
    Conv24 = Conv2D(filters=64, kernel_size=(kernel,kernel), padding='same', activation=ActFunction, data_format='channels_last', name='Conv24')(Act23)
    Norm24 = BatchNormalization(name='Norm24')(Conv24)
    Act24 = Activation(ActFunction, name=ActFunction+'24')(Norm24)

    UnPool1 = MaxUnpooling2D(name='Unpool1')([Act24, Argmax1], tf.shape(Conv2))
    Conv25 = Conv2D(filters=64, kernel_size=(kernel,kernel), padding='same', activation=ActFunction, data_format='channels_last', name='Conv25')(UnPool1)
    Norm25 = BatchNormalization(name='Norm25')(Conv25)
    Act25 = Activation(ActFunction, name=ActFunction+'25')(Norm25)
    Conv26 = Conv2D(filters=out_channels, kernel_size=(kernel,kernel), padding='same', activation=out_ActFunction, data_format='channels_last', name='Conv26')(Act25)
    Norm26 = BatchNormalization(name='Norm26')(Conv26)
    
    Out = Conv2D(out_channels, 1, activation = out_ActFunction, name='OutputLayer')(Norm26)
    return tf.keras.Model(inputs = Input, outputs = Out, name = name)