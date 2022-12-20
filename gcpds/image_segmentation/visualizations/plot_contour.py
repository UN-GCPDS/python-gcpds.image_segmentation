import numpy as np
from typing import Tuple, List 
import matplotlib.pyplot as plt

Mask = Tuple[np.ndarray, str, str]


def plot_contour(img: np.ndarray, masks: List[Mask],
                  ax: plt.axes) -> plt.axes:
    """ Plot multiple contours over image.

    Parameters
    ----------
    img :
        Image RGB or Gray scale
    masks :
        List of tuples mask, name and color. Mask in 2D.
    ax :
        matplotlib ax
        
    Returns
    -------
    ax :  
        matplotlib ax
    """
    ax.imshow(img)
    h = []
    names = []
    for mask, name, color in masks:
        if not np.any(mask):
            continue
        cntr = ax.contour(mask, levels=[0.5], colors=color)
        handler, _ = cntr.legend_elements()
        h.append(handler[0])
        names.append(name)

    ax.legend(h,names)
    ax.axis('off')
    return ax 
        

