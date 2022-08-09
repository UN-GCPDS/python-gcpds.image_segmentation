""" 
Fusion cam from multiples layers 
References http://mftp.mmcheng.net/Papers/21TIP_LayerCAM.pdf
"""


import numpy as np 
from tqdm import tqdm 
import numpy as np 
from typing import Callable, Iterable

from tensorflow.math import softmax


def _cumulative_maximun() -> Callable:
    """ Accumulation of the maximum
    """

    has_started = False 
    result = None 

    def maximun(input_ : np.array)  -> np.array:
        nonlocal has_started, result
        
        if not has_started:
            has_started = True 
            result = input_

        result = result[...,None]
        input_ = input_[...,None]

        concat = np. concatenate([result,input_],axis=-1)
        result = np.max(concat, axis=-1)
        return result

    return maximun


def _cumulative_weights(return_weights) -> Callable:
    """ Accumulation for computing wiegths per layer
    """

    has_started = False 
    result = None 
    cams = []
    weights = []

    def weights_(input_ : np.array)  -> np.array:
        nonlocal has_started, result, cams
        
        wiegth = np.sum(input_, axis=(-2,-1))
        cams.append(input_[None,...])
        
        weights.append(wiegth)
        result = softmax(np.vstack(weights),axis=0)[...,None,None]*np.vstack(cams)

        if return_weights:
            return np.sum(result, axis=0), np.vstack(weights)
        else: 
            return np.sum(result, axis=0)

    return weights_



def fusion_cam(callable_cam : Callable, images : np.array, 
                score_function : Callable, layers : Iterable,
                type_fusion = 'maximun', return_weights=False) -> np.array:

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
    type_fusion:
        Type fusion ['maximun', 'weights']
    return_weights:
        return weights of type weights fusion
    Returns
    -------
    np.array 
        Result fusion of the CAMs at multiple layers.
    
    """

    if type_fusion == 'maximun':
        aggregation = _cumulative_maximun()
    elif type_fusion == 'weights':
        aggregation = _cumulative_weights(return_weights)

    layers = tqdm(layers)
    for layer in layers:
        layers.set_postfix({'Layer ':layer})
        cam = callable_cam(score_function, images, penultimate_layer=layer,
                            seek_penultimate_conv_layer=False)
        
        if type(cam) is list:
            cam = [aggregation(i) for i in cam]
        else:
            cam = aggregation(cam)
    if return_weights:
        return cam[0],cam[1]
    else:
        return cam 