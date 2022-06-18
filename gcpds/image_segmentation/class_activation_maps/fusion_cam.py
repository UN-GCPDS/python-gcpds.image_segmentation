""" 
Fusion cam from multiples layers 

References http://mftp.mmcheng.net/Papers/21TIP_LayerCAM.pdf
"""


import numpy as np 
from tqdm import tqdm 
import numpy as np 
from typing import Callable, Iterable


def __cumulative_maximun() -> Callable:
    """ Accumulation of the maximum
    """

    flag_start = False 
    result = None 

    def maximun(input_ : np.array)  -> np.array:
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


def fusion_cam(callable_cam : Callable, images : np.array, 
                score_function : Callable, layers : Iterable) -> np.array:

    """ Fusion CAM from multiple layers without scaling just maximun.

    Parameters
    ----------
    callable_cam :
        Function to obtain CAM.
    images :
        Images to obtain CAM.
    score_function :
        Same score function used to calculate the CAMs in tf-keras-vis.
    layers :
        Layers where obtain the CAMs.

    Returns
    -------
    np.array 
        Result fusion of the CAMs at multiple layers.
    
    """
    maximun = __cumulative_maximun()

    layers = tqdm(layers)
    for layer in layers:
        layers.set_postfix({'Layer ':layer})
        cam = callable_cam(score_function, images, penultimate_layer=layer,
                            seek_penultimate_conv_layer=False)
        cam = maximun(cam)
    
    return cam 