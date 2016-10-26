import argparse

def bl832inputs_parser(parser):
        #Parameters associated with the acquired data          
   
        parser.add_argument("--input_hdf5", help="Full path of the input hdf5 file")
        parser.add_argument("--group_hdf5", help="Full path of the input group to reconstruct")
        parser.add_argument("--output_hdf5", help="Full path of the output hdf5 file")
        parser.add_argument("--rot_center", help="Center of rotation of the object in units of detector pixels",type=float,default=-1)
        parser.add_argument("--pix_size", help="Size of the detector pixels in micro meter",type=float)
        parser.add_argument("--num_views", help="Number of views for the data",type=int)
        parser.add_argument("--num_bright",help="Number of bright fields acquired prior the the experiment",type=int)
        parser.add_argument("--num_dark",help="Number of dark fields acquired prior the the experiment",type=int)
        parser.add_argument("--inter_bright",help="Use intermediate brights for normalization. Default is to use brights from end only.",action="store_true")
        parser.add_argument("--dual_norm",help="Average brights and darks from the beginning AND end.",action="store_true")
        parser.add_argument("--full_rot",help="Use this flag if the views correspond to 0-360 degrees instead of 0-180 degrees.",action="store_true")
        parser.add_argument("--view_subsmpl_fact", help="View skipping factor. This can be used to subset the data in terms of views",type=int,default=1)
        parser.add_argument("--gpu_device", help="Device id of the GPU",type=int,default=0)

        #Reconstruction parameters                                                                                                                                        
        parser.add_argument("--x_width", help="Number of detector elements along x-direction",type=int)
        parser.add_argument("--z_start", help="Starting detector pixel along z-direction",type=int)
        parser.add_argument("--z_numElts", help="Number of detector pixels to use along z-direction",type=int)
        parser.add_argument("--p",help="qGGMRF prior model parameter use to control how smooth the edge smoothness",type=float,default=1.2)
        parser.add_argument("--smoothness",help="A scaling parameter use to control the balance between resolution and noise in the reconstruction. This value is multiplied by a predtermined internal value",type=float,default=0.50)
        parser.add_argument("--zinger_thresh",help="Controls the rejection thresold of bad measurements that cause zingers. In a future version this will have proper units\
 At present try values in the range 1-50 to correct for zingers",type=float,default=10000)
        parser.add_argument("--filter_param",help="Filter cut-off (0-1)",type=float,default=1)

        #Advanced parameters which the user need not worry about but can manipulate if necessary                                                                            
        parser.add_argument("--stop_threshold",help="Stopping thresold as a percentage of average change in pixel values in percentage",type=float,default=10)
        parser.add_argument("--max_iter",help="Maximum number of ICD iterations for the algorithm",type=int,default=30)

        args = parser.parse_args()

        inputs = {}
        inputs['input_hdf5'] = args.input_hdf5
        inputs['group_hdf5'] = args.group_hdf5
        #The group name has to start with a "/"                                       
        if (inputs['group_hdf5'][0] != '/'):
            print 'Group name has to start with a /'
            return -1

        inputs['output_hdf5']= args.output_hdf5
        inputs['gpu_device'] = args.gpu_device

        inputs['pix_size'] = args.pix_size
        inputs['num_bright'] = args.num_bright
        inputs['num_dark'] = args.num_dark
        inputs['num_views'] = args.num_views
        inputs['view_subsmpl_fact'] = args.view_subsmpl_fact
        inputs['x_width'] = args.x_width
        inputs['rot_center'] = args.rot_center
        inputs['z_start'] = args.z_start
        inputs['z_numElts'] = args.z_numElts
        inputs['p'] = args.p
        inputs['fbp_filter_param']=args.filter_param
        inputs['smoothness'] = args.smoothness
        inputs['stop_threshold'] = args.stop_threshold
        inputs['max_iter'] = args.max_iter
        inputs['inter_bright'] = args.inter_bright                                   
        inputs['dual_norm'] = args.dual_norm  
        inputs['full_rot'] = args.full_rot  
        inputs['num_views'] = args.num_views
        return inputs
