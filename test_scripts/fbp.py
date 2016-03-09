#script to test fbp 

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale, iradon

image = imread(data_dir + "/phantom.png", as_grey=True)
image = rescale(image, scale=0.4)
plt.imshow(image,cmap=plt.cm.Greys_r)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4.5))

ax1.set_title("Original")
ax1.imshow(image, cmap=plt.cm.Greys_r)

theta = np.linspace(0., 180., max(image.shape), endpoint=False)
sinogram = radon(image, theta=theta, circle=True)
ax2.set_title("Radon transform\n(Sinogram)")
ax2.set_xlabel("Projection angle (deg)")
ax2.set_ylabel("Projection position (pixels)")
ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
           extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

recon = iradon(sinogram,theta,160,filter='hamming',interpolation='linear')
#plt.subplot(224)
ax3.set_title("Reconstruction\nfrom sinogram")
ax3.imshow(recon, cmap=plt.cm.Greys_r)

fig.tight_layout()
plt.show()
