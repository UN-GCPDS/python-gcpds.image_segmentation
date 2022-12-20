import sys
sys.path.append("../../")

import matplotlib.pyplot as plt 
import numpy as np

from gcpds.image_segmentation.visualizations import plot_contour
from gcpds.image_segmentation.datasets.segmentation import NerveUtp

dataset = NerveUtp(split=[0.64,0.16,0.2],seed=70)
train_dataset, test_dataset, val_dataset = dataset()
train_dataset = train_dataset.batch(1)
test_dataset = test_dataset.batch(1)
val_dataset = val_dataset.batch(1)

print("Nerves: ", dataset.labels_info)
for img, mask, label, id_img in train_dataset.take(1):
    fig, ax = plt.subplots()
    img = np.squeeze(img)
    mask =  np.squeeze(mask)
    masks = [(mask,'maks1','red'),(mask*0,'mask2','green')]
    plot_contour(img, masks, ax=ax)

plt.show()