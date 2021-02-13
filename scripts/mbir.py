import numpy as np
from tomocam import MBIR
import tomopy

import matplotlib.pyplot as plt

center = 200.
shepp = tomopy.misc.phantom.shepp2d(size = 400, dtype=np.float32)
angles = np.linspace(0, np.pi, 360, dtype=np.float32)
tomo = tomopy.sim.project.project(shepp, angles, center = center, pad = False, sinogram_order=True)

recon = MBIR(tomo.astype(np.float32), angles, center, num_iters = 150, smoothness = 10)

plt.imshow(recon[0])
plt.show()

