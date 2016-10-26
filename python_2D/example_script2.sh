#!/bin/bash

input_file="/home/svvenkatakrishnan/data/20131106_074854_S45L3_notch_OP_10x.h5"
#20160717_164333_NASA_FiberInCell_TrajectoryD.h5"
#20130807_234356_OIM121R_SAXS_5x.h5"
group_h5="20131106_074854_S45L3_notch_OP_10x"
#"20160717_164333_NASA_FiberInCell_TrajectoryD"
#"20130807_234356_OIM121R_SAXS_5x"
center=1328 #982.5 #1294
pix_size=1.3
num_views=1024 #512 
num_brt=30 #15 
num_dark=10
view_sub=1
xwidth=2560 #2000 
z_start=0
z_numElts=132
curr_slice=0

#gpu_device=0

for gpu_device in 0 1 2 3 4 5 6 7 
do
    let "curr_slice=$z_start+$gpu_device*$z_numElts"
#    echo $curr_slice
    python reconstruct_gridrec.py --input_hdf5 $input_file --group_hdf5 /$group_h5 --output_hdf5 temp.h5 --rot_center $center --pix_size $pix_size --num_views $num_views --num_bright $num_brt --num_dark $num_dark --view_subsmpl_fact $view_sub --x_width $xwidth --z_start $curr_slice --z_numElts $z_numElts --p 1.2 --smoothness 10 --zinger_thresh 0 --filter_param 0.5 --gpu_device $gpu_device &
done