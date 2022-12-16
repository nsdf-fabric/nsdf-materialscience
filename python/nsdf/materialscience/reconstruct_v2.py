"""
http://nationalsciencedatafabric.org

This material is based upon work supported by the National Science Foundation under Grant No. 2138811. 
Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) 
and do not necessarily reflect the views of the National Science Foundation. Copyright © 2021 National Science Data Fabric
"""

import os,gc,sys,time
import numpy as np
import h5py
import astra
import tomopy
import dxchange
from   ccpi.filters.regularisers import PD_TV

# /////////////////////////////////////////////////////////////
# 8 bit Grayscale Image (Pixel range: 0-255)
def convert_to_8bit(img, vmin, vmax):
    result = np.clip(img.astype(float) - vmin, 0, vmax-vmin) / (vmax - vmin) 
    return result * 255.0

# /////////////////////////////////////////////////////////////
# 16 bit Grayscale Image (Pixel range: 0-65,535)
def convert_to_16bit(img, vmin, vmax):
    result = np.clip(img.astype(float) - vmin, 0, vmax-vmin) / (vmax - vmin) 
    return result * (2**16-1)

# /////////////////////////////////////////////////////////////
def reconstruct_image(input_filename, output_filename, ind_start, ind_end, rot_cen, white_universal, write_tiff_digit=5):
    """
    Read the radiographs, flat-field & dark-field images from the h5 file
    """

    Data = h5py.File(input_filename, 'r')
    
    # Load the Angle Information
    projections = Data['img_tomo']
    theta = np.array(Data['angle']) * np.pi / 180
    
    # Mask the obstructed projections
    idx_angle_filter = projections.shape[1]//2
    threshold = projections[:, idx_angle_filter].mean(axis=0)
    theta_filter = (projections[:, idx_angle_filter] > threshold).mean(axis=-1) > 0.5
    theta_slice = np.arange(theta.size, dtype='int')[theta_filter]
    theta_slice = slice(theta_slice.min(), theta_slice.max())
    theta = theta[theta_slice]
    del projections  # deleted the projections to save the memory space
    
    SLICE_RANGE = slice(ind_start,ind_end)
    # Load the White Field Images
    white = Data['img_bkg'][:,SLICE_RANGE]
        
    # Load the Dark Field Images
    dark = Data['img_dark'][:,SLICE_RANGE]
    
    if white.mean() < 2 * dark.mean(): 
        white = white_universal[:,SLICE_RANGE]
    else:
        white = white # then we can use the original image

    # Load the Projections/Radiographs
    projections = Data['img_tomo'][theta_slice,SLICE_RANGE]
    
    #Perform the flat-field correction of raw data:
    projections_norm = tomopy.normalize(projections, white[::2,:], dark[::2,:], ncore = 16)
    
    # Calculate $ -log(proj) $ to linearize transmission tomography data.
    projections_neglog = tomopy.minus_log(projections_norm, ncore = 16)
    projections_neglog = tomopy.remove_nan(projections_neglog,ncore = 16)
    
    # Stripe Removal Process
    projections_strprem = tomopy.remove_all_stripe(projections_neglog,
                                                   snr=3,
                                                   la_size=61,
                                                   sm_size=41,
                                                   ncore=16)

    # Phase retrival operation
    projections_phsret = tomopy.retrieve_phase(projections_strprem,
                                               pixel_size=6.49/10000,
                                               dist=2.5,
                                               energy=5,
                                               alpha = 0.01,
                                               pad='True',
                                               ncore=16)


    #Setup ASTRA options
    opts = {}
    opts['method'] = 'CGLS_CUDA'
    opts['num_iter'] = 5
    opts['proj_type'] = 'cuda'
    #opts['extra_options'] = {'MinConstraint':0}
    #opts['gpu_list'] = [0, 1]

    ReconDataAstra = tomopy.recon(projections_phsret,
                                  theta,
                                  center=rot_cen,
                                  algorithm= tomopy.astra,
                                  options=opts,
                                  ncore=1)

    # set parameters
    pars = {
        'algorithm' : PD_TV, 
        'input' : ReconDataAstra,
        'regularisation_parameter':0.02,
        'number_of_iterations' :5 ,
        'tolerance_constant':1e-06,
        'methodTV': 0 ,
        'nonneg': 0,
        'lipschitz_const' : 8
    }  

    (pd_gpu3D, info_vec_gpu)  = PD_TV(
        pars['input'], 
        pars['regularisation_parameter'],
        pars['number_of_iterations'],
        pars['tolerance_constant'], 
        pars['methodTV'],
        pars['nonneg'],
        pars['lipschitz_const'],'gpu'
    )

    # Export reconstructed slices as TIFFs
    vmin, vmax = np.percentile(pd_gpu3D,(0.1, 99.9))
    img_to_save = convert_to_16bit(tomopy.circ_mask(pd_gpu3D,axis=0, ratio=0.85), vmin, vmax)
    os.makedirs(os.path.dirname(output_filename),exist_ok=True)
    dxchange.writer.write_tiff_stack(img_to_save, fname = output_filename , axis=0, start=ind_start, overwrite=True,digit=write_tiff_digit)
    return vmin,vmax

# /////////////////////////////////////////////////////////////////
### Main Script ###
if __name__=="__main__":

	### Main Script ###
	center_shift_dict = {  # Calculated center shift values for each file.
		"fly_scan_id_112437": 1273,
		"fly_scan_id_112441": 1278,
		"fly_scan_id_112443": 1277,
		"fly_scan_id_112451": 1272.50,
		"fly_scan_id_112453": 1273.50,
		"fly_scan_id_112455": 1278,
		"fly_scan_id_112458": 1278,
		"fly_scan_id_112461": 1278.50,
		"fly_scan_id_112464": 1276.50,
		"fly_scan_id_112467": 1278,
		"fly_scan_id_112470": 1275.50,
		"fly_scan_id_112475": 1276,
		"fly_scan_id_112478": 1276.50,
		"fly_scan_id_112482": 1277.50,
		"fly_scan_id_112484": 1273,
		"fly_scan_id_112487": 1278.50,
		"fly_scan_id_112489": 1278.50,
		"fly_scan_id_112491": 1276,
		"fly_scan_id_112493": 1276.50,
		"fly_scan_id_112495": 1277.50,
		"fly_scan_id_112500": 1278,
		"fly_scan_id_112502": 1278,
		"fly_scan_id_112504": 1277,
	#     "fly_scan_id_112509": 1270,
	#     "fly_scan_id_112512": 1273,
	#     "fly_scan_id_112515": 1271.50,
	#     "fly_scan_id_112517": 1265,
	#     "fly_scan_id_112520": 1266.50,
	#     "fly_scan_id_112522": 1266.50,
	#     "fly_scan_id_112524": 1274,
	#     "fly_scan_id_112526": 1267.50,
	#     "fly_scan_id_112528": 1271,
	#     "fly_scan_id_112530": 1267.50,
	#     "fly_scan_id_112532": 1272,
	#     "fly_scan_id_112545": 1250,
	#     "fly_scan_id_112548": 1250,
	#     "fly_scan_id_112550": 1249.50,
	#     "fly_scan_id_112552": 1255.50,
	#     "fly_scan_id_112554": 1252,
	#     "fly_scan_id_112556": 1256.50,
	#     "fly_scan_id_112558": 1258.50,
	#     "fly_scan_id_112560": 1259,
	#     "fly_scan_id_112577": 1266,
	#     "fly_scan_id_112579": 1266,
	#     "fly_scan_id_112581": 1260.5,
	#     "fly_scan_id_112583": 1266,
	#     "fly_scan_id_112585": 1262.5,
	#     "fly_scan_id_112587": 1264,
	#     "fly_scan_id_112589": 1258,
	#     "fly_scan_id_112591": 1266.5,
	#     "fly_scan_id_112593": 1266,
	#     "fly_scan_id_112595": 1260,
	#     "fly_scan_id_112597": 1267,
	#     "fly_scan_id_112599": 1266,
	#     "fly_scan_id_112601": 1250,
	#     "fly_scan_id_112603": 1254,
		}

	
	# Read the radiographs, flat-field & dark-field images from the h5 file
	SRC_path = '.'
	DST_path = 'Reconstructions'
	filename = 'fly_scan_id_112437'
	input_filename = os.path.join(SRC_path, filename + '.h5')
	output_filename = os.path.join(DST_path, filename, 'Reconstructions')
	rot_cen = center_shift_dict[filename]

	# Get the Slice Limit Information
	Data = h5py.File(os.path.join(SRC_path, filename + '.h5'), 'r')
	projections = Data['img_tomo']
	Slice_limits = projections.shape[1]
	del projections
	del Data

	# Load a universal "white" image:
	Data_Universal = h5py.File(os.path.join(SRC_path, 'fly_scan_id_112509' + '.h5'), 'r')
	white_universal = Data_Universal['img_bkg']
	del Data_Universal

	if False:

		out = h5py.File("temp.hdf5", "w")
		out.create_dataset('img_bkg', data=white_universal)
		out.close()

		Data_Universal = h5py.File("temp.hdf5", 'r')
		white_universal = Data_Universal['img_bkg']
		del Data_Universal

	for ind_start in range(1020,1030, 5):
		ind_end = ind_start+5
		print("Reconstructing offset",ind_start)
		reconstruct_image(input_filename, output_filename, ind_start, ind_end, rot_cen, white_universal, write_tiff_digit=5)
