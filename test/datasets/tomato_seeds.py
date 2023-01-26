import sys
sys.path.append("../../")

import matplotlib.pyplot as plt 

from gcpds.image_segmentation.datasets.segmentation import TomatoSeeds

dataset = TomatoSeeds(split=[0.64,0.16,0.2],seed=70)
train_dataset, test_dataset, val_dataset = dataset()
train_dataset = train_dataset.batch(1)
test_dataset = test_dataset.batch(1)
val_dataset = val_dataset.batch(1)

for img, mask, id_img in train_dataset.take(1):
    plt.subplot(1,2,1)
    plt.imshow(img[0,...])
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(mask[0,...])
    plt.colorbar()

plt.show()
