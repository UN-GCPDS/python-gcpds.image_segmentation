""" 
Fusion cam from multiples layers 

References http://mftp.mmcheng.net/Papers/21TIP_LayerCAM.pdf
"""


import numpy as np 
from tqdm import tqdm 


def __cumulative_maximun():
    flag_start = False 
    result = None 

    def maximun(input_):
        nonlocal flag_start, result
        
        if not flag_start:
            flag_start = True 
            result = input_

        result = result[...,None]
        input_ = input_[...,None]

        concat = np. concatenate([result,input_],axis=-1)
        result = np.max(concat, axis=-1)
        return result

    return maximun


def fusion_cam(callable_cam, images, score, layers):
    maximun = __cumulative_maximun()

    layers = tqdm(layers)
    for layer in layers:
        layers.set_postfix({'Layer ':layer})
        cam = callable_cam(score, images, penultimate_layer=layer,
                            seek_penultimate_conv_layer=False)
        cam = maximun(cam)
    
    return cam 