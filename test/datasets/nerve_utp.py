import sys
sys.path.append("../../")

import matplotlib.pyplot as plt 

from gcpds.image_segmentation.datasets.segmentation import NerveUtp

dataset = NerveUtp()
dataset = dataset()
dataset = dataset.batch(1)

for img, mask, label in dataset.take(1):
    plt.subplot(1,2,1)
    plt.imshow(img[0,...])
    plt.subplot(1,2,2)
    plt.imshow(mask[0,...,0])
    plt.suptitle(str(label))

plt.show()
