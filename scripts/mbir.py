import numpy as np
import tomocam



# read sinogram data
data = np.fromfile('/home/dkumar/Data/phantom/sino_00016.bin', dtype=np.float32)
sino = tomocam.DistArray(data.reshape(20, 1536, 1280))
model = tomocam.DistArray(np.ones((20, 1280, 1280), dtype=np.float32))
angles = np.linspace(0, np.pi, sino.shape[1], dtype = np.float32)

n_iters = 10
for i in range(n_iters):
    gradients = model.copy()
    print('copied ....')
    tomocam.calc_gradients(gradients, sino, angles, center=640)
    print('gradients done ....')
    tomocam.update_total_variation(model, gradients)
    print('tv done ....')
    tomocam.axpy(-0.1, gradients, model)
    print('update done ...')
    error = 0.1 * gradients.norm()
    print(error)
