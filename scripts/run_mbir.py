#script to test tomopy

import numpy as np
import json
import time
import sys
import matplotlib.pyplot as plt
from pathlib import Path

import tomopy
import tomocam
import dxchange as dx



if __name__ == '__main__':
    
    # parse json input
    if len(sys.argv) < 2:
        print("Usage: python run_mbir.py <input.json>")
        exit()

    json_file = sys.argv[1]
    with open(json_file, 'r') as f:
        inp = json.load(f)

    datadir = inp['input_dir']
    filename = inp['input_file']
    outputdir = inp['output_dir']
    axis = inp['axis']
    if "MBIR" in inp:
        num_iters = inp["MBIR"]["num_iters"]
        smoothness = inp["MBIR"]["smoothness"]
        if "tol" in inp["MBIR"]:
            tol = inp["MBIR"]["tol"]
        else: 
            tol = 1e-4
        if "xtol" in inp["MBIR"]:
            xtol = inp["MBIR"]["xtol"]
        else:
            xtol = 1e-4
    else:
        num_iters = 100
        smoothness = 0.01
        tol = 1e-4
        xtol = 1e-4
    
    # print input data 
    print("Input data:")
    print("Data file: ", datadir + '/' + filename)
    print("axis: ", axis)
    print("num_iters: ", num_iters)
    print("smoothness: ", smoothness)
    print("tol: ", tol)
    print("xtol: ", xtol)

    dataset = Path(datadir) / filename
    if not dataset.exists():
        raise FileNotFoundError(f"File {dataset} not found")

    tomo, flat, dark, theta = dx.read_aps_32id(dataset,sino=(1000, 1016, 1))

    tomo = tomo.astype(np.float32)
    theta = theta.astype(np.float32)

    tomo = tomopy.normalize(tomo, flat, dark, out=tomo)

    mx = np.float32(0.01)
    tomo[tomo < 0.01] = 0.01
    tomo = tomopy.minus_log(tomo)
    tomo = tomopy.remove_stripe_fw(tomo)

    t0 = time.time()
    tomo = np.transpose(tomo, (1, 0, 2))
    rec = tomocam.MBIR(tomo, theta, center=axis, num_iters=num_iters, smoothness=smoothness, tol=tol, xtol=xtol)
    rec = tomopy.circ_mask(rec, axis=0, ratio=1.0)
    print(time.time() - t0)

    # plot reconstruction
    plt.imshow(rec[0], cmap='Greys_r')
    plt.show()
