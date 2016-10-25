#!/bin/bash

input_file="/home/svvenkatakrishnan/data/20130807_234356_OIM121R_SAXS_5x.h5"
group_h5="20130807_234356_OIM121R_SAXS_5x"
center=1294
pix_size=1.3
num_views=1024
num_brt=30
num_dark=10
view_sub=1
xwidth=2560
z_start=1600
z_numElts=130
gpu_device=3

python reconstruct_gridrec.py --input_hdf5 $input_file --group_hdf5 /$group_h5 --output_hdf5 temp.h5 --rot_center $center --pix_size $pix_size --num_views $num_views --num_bright $num_brt --num_dark $num_dark --view_subsmpl_fact $view_sub --x_width $xwidth --z_start $z_start --z_numElts $z_numElts --p 1.2 --smoothness 10 --zinger_thresh 0 --gpu_device $gpu_device
