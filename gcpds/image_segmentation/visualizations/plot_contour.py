"""
============
Plot Contour
============
"""


import numpy as np
from typing import Tuple, List 
import matplotlib.pyplot as plt

Mask = Tuple[np.ndarray, str, str]


def plot_contour(img: np.ndarray, masks: List[Mask],
                  ax: plt.axes, cmap: str='gray') -> plt.axes:
    """ Plot multiple contours over image.

    Parameters
    ----------
    img :
        Image RGB or Gray scale
    masks :
        List of tuples mask, name and color. Mask in 2D.
    ax :
        matplotlib ax
    cmap:
        cmap for imshow
        
    Returns
    -------
    ax :  
        matplotlib ax
    """
    if np.any(img):
        ax.imshow(img, cmap=cmap)
    h = []
    l = []
    for mask, name, color in masks:
        if not np.any(mask):
            continue
        cntr = ax.contour(mask, levels=[0.5], colors=color)
        handler, _ = cntr.legend_elements()
        h.append(handler[0])
        l.append(name)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
    
    ax.legend(h,l, loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=len(l))
    ax.axis('off')
    
    return ax 