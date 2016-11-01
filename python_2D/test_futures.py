#Code by Daniel Pelt (Oct 2016)
#Modifications by S. V. Venkatakishnan

from __future__ import print_function
import numpy as np
import concurrent.futures as cf
import afnumpy as afnp
import arrayfire as af
import time

def add(a,b,c):
    c[:] = a+b

def gpu_add(a,b,c,dev_id):
    af.set_device(dev_id)
    ga=afnp.array(a)
    gb=afnp.array(b)
    gc=ga+gb
    c[:] = np.array(gc)

def gpu_add_v2(a,b,c,i,blk_size):
    af.set_device(i)
    ga=afnp.array(a[i*blk_size:(i+1)*blk_size])
    gb=afnp.array(b[i*blk_size:(i+1)*blk_size])
    gc=ga+gb
    c = np.array(gc)

num_gpu = 8
arr_size = 512
a = np.array(np.random.rand(arr_size, 256,256),dtype=np.float32)
b = np.array(np.random.rand(arr_size, 256,256),dtype=np.float32)
c = np.array(np.random.rand(arr_size, 256,256),dtype=np.float32)
#c = np.array(np.random.rand(num_gpu,arr_size/num_gpu, 2560,2560),dtype=np.float32)

blk_size = arr_size/num_gpu
e = cf.ThreadPoolExecutor(num_gpu)
print('Starting compute ..')
t=time.time()
for i in range(num_gpu):
#  e.submit(add, a[i*blk_size:(i+1)*blk_size], b[i*blk_size:(i+1)*blk_size], c[i*blk_size:(i+1)*blk_size])
  e.submit(gpu_add, a[i*blk_size:(i+1)*blk_size], b[i*blk_size:(i+1)*blk_size], c[i*blk_size:(i+1)*blk_size],i)
#  e.submit(gpu_add_v2, a, b, c[i],i,blk_size)
e.shutdown()
elapsed_time = (time.time()-t)
print('Time for addition : %f' % elapsed_time)
print(np.allclose(a+b, c))
