import sys
sys.path.append("../../")

import matplotlib.pyplot as plt 

from gcpds.image_segmentation.datasets.segmentation import NerveUtp

dataset = NerveUtp(split=0.2)
train_dataset, test_dataset = dataset()
train_dataset, test_dataset = train_dataset.batch(1), test_dataset.batch(1)

for img, mask, label in train_dataset.take(1):
    plt.subplot(1,2,1)
    plt.imshow(img[0,...])
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(mask[0,...,0])
    plt.colorbar()
    plt.suptitle(str(label))

plt.show()
