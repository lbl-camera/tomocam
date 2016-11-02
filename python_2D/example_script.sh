!/bin/bash 


for slices in 100 #12 24 48 96
do
  for sub_samp in 1 #1 2 4
  do
      python reconstruct_gridrec.py --input_hdf5 ~/data/20130807_234356_OIM121R_SAXS_5x.h5 --group_hdf5 /20130807_234356_OIM121R_SAXS_5x --output_hdf5 temp.h5 --rot_center 1280 --pix_size 1.3 --num_views 1024 --num_bright 30 --num_dark 10 --view_subsmpl_fact $sub_samp --x_width 2560 --z_start 1900 --z_numElts $slices --p 1.2 --smoothness 10 --zinger_thresh 0

#    python reconstruct_gridrec.py --input_hdf5 ~/data/20160717_164333_NASA_FiberInCell_TrajectoryD.h5 --group_hdf5 /20160717_164333_NASA_FiberInCell_TrajectoryD --output_hdf5 temp.h5 --rot_center 982.5 --pix_size 5.3 --num_views 512 --num_bright 15 --num_dark 10 --view_subsmpl_fact $sub_samp --x_width 2000 --z_start 100 --z_numElts $slices --p 1.2 --smoothness 10 --zinger_thresh 0
#      python reconstruct_gridrec.py --input_hdf5 ~/data/20131106_074854_S45L3_notch_OP_10x.h5 --group_hdf5 /20131106_074854_S45L3_notch_OP_10x --output_hdf5 temp.h5 --rot_center 1328 --pix_size 1.3 --num_views 1024 --num_bright 30 --num_dark 10 --view_subsmpl_fact $sub_samp --x_width 2560 --z_start 1000 --z_numElts $slices --p 1.2 --smoothness 10 --zinger_thresh 0      
  done
done
