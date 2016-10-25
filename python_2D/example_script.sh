!/bin/bash 

#python main.py --input_hdf5 ~/data/20130807_234356_OIM121R_SAXS_5x.h5 --group_hdf5 /20130807_234356_OIM121R_SAXS_5x --output_hdf5 temp.h5 --rot_center 1280 --pix_size 1.3 --num_views 1024 --num_bright 30 --num_dark 10 --view_subsmpl_fact 4 --x_width 2560 --z_start 1000 --z_numElts 3 --p 1.2 --smoothness 10 --zinger_thresh 0

python main.py --input_hdf5 ~/data/20131106_074854_S45L3_notch_OP_10x.h5 --group_hdf5 /20131106_074854_S45L3_notch_OP_10x --output_hdf5 temp.h5 --rot_center 1328 --pix_size 1.3 --num_views 1024 --num_bright 30 --num_dark 10 --view_subsmpl_fact 1 --x_width 2560 --z_start 1000 --z_numElts 100 --p 1.2 --smoothness 10 --zinger_thresh 0
