% test_all_fbp

list = {
 'ct_geom test'
 'image_geom test'
 'sino_geom test'
 'cylinder_proj test'
 'df_example1'
 'ellipse_im test'
 'ellipse_sino test'
 'ellipsoid_proj test'
 'ellipsoid_im test'
%'fan_rebin'
 'fbp_fan_arc_example'
 'fbp_fan_arc_point'
 'fbp_fan_flat_example'
 'fbp_ramp test'
 'fbp2_sino_filter test'
 'fbp2_example'
 'feldkamp_example'
 'jaszczak1 test'
%'sphere_proj test'
};

im nan-fail
run_mfile_local(list)
