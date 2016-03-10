import matplotlib.pyplot as plt
import tomopy
from normalize import normalize_bo
import argparse
from XT_argsParser import bl832inputs_parser
import time
import os
import math


def main():

        parser = argparse.ArgumentParser()
        inputs = bl832inputs_parser(parser)
        print "Input file :",inputs['input_hdf5']

        algorithm='fbp'
        tomo, flats, darks, floc = tomopy.read_als_832h5(inputs['input_hdf5'],sino=(1000, 1003, 1))
        print('Data read complete')
        print tomo.shape

        print('Displaying  sinogram')
#        imgplot = plt.imshow(tomo[:,0,:])

        print('Generating angles')
        theta = tomopy.angles(tomo.shape[0])

        print('Normalization')
        tomo = normalize_bo(tomo, flats, darks,20)

        print('Ring removal')
        tomo = tomopy.remove_stripe_fw(tomo)

        print('Recon')
        rec = tomopy.recon(tomo, theta, center=1294,algorithm='osem',emission=False)

        print('Masking')
        rec = tomopy.circ_mask(rec, 0)

        tomopy.write_tiff_stack(rec, 'test.tiff')
        print 'main: Done!'

main()


