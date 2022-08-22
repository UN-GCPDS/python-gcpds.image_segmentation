import sys
sys.path.append("../../")

import matplotlib.pyplot as plt 
from gcpds.image_segmentation.datasets.segmentation import SinteticDataset


samples = 6
dataset = SinteticDataset(samples=samples,img_shape=128,seed=32)
dataset = dataset()
dataset = dataset.batch(1)

for i,(img, mask) in enumerate(dataset):
    plt.subplot(2,samples,2*(i+1)-1)
    plt.imshow(img[0,...])
    plt.axis('off')
    plt.title('Image')

    plt.subplot(2,samples,2*(i+1))
    plt.imshow(mask[0,...,0])
    plt.axis('off')
    plt.title('Mask')

plt.show()




