import matplotlib.pylab as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import filters
from skimage import exposure

import os,sys,glob
import cv2 
import numpy as np
import multiprocessing
import tifffile

# ////////////////////////////////////////////////////////////////
def GetImageInfo(img):
	return dict(shape=img.shape,dtype=img.dtype,m=np.min(img),M=np.max(img))

def ScaleImage(img,a,b):
	m,M=np.min(img),np.max(img)
	return a+b*((img-m)/(M-m))

import glob
import numpy as np
from matplotlib import pyplot

from skimage import filters
from skimage.data import camera
from skimage.util import compare_images



# https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html

sobel_x=[
  [ -1 , 0 ,  +1] , 
  [-2 , 0 , +2] , 
  [ -1 , 0 ,  +1] ]
sobel_y=[
  [-1, -2, -1] , 
  [ 0,  0,  0] , 
  [+1, +2, +1] ]

sharr_x=[
  [ -3 , 0 ,  +3] , 
  [-10 , 0 , +10] , 
  [ -3 , 0 ,  +3] ]
sharr_y=[
  [-3, -10, -3] , 
  [ 0,   0,  0] , 
  [+3, +10, +3] ]


DIR="C:/data/visus-datasets/Pania_2021Q3_in_situ_data/EFRC_Related_Data_For_External_Collaboration/Radiographic_Scans_Tiffs"

sequence=[
	"radiographic_scan_id_112510",
	"radiographic_scan_id_112511",
	"radiographic_scan_id_112513",
	"radiographic_scan_id_112514",
	"radiographic_scan_id_112516",
	"radiographic_scan_id_112518",
	"radiographic_scan_id_112519",
	"radiographic_scan_id_112521",
	"radiographic_scan_id_112523",
	"radiographic_scan_id_112525",
	"radiographic_scan_id_112527",
	"radiographic_scan_id_112529",
	"radiographic_scan_id_112531",
	"radiographic_scan_id_112533",
]

for seq in sequence:
	  
	for in_filename in sorted(glob.glob(f"{DIR}/{it}/*.tiff")):
		out_filename=f"C:/data/visus-datasets/Pania_2021Q3_in_situ_data/EFRC_Related_Data_For_External_Collaboration/animation/{seq}_{os.path.basename(in_filename)}"
		print("Doing",in_filename,out_filename)

		img=tifffile.imread(in_filename);assert img.dtype==np.uint16
		img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
		img = img/65536.0

		dx= cv2.filter2D(img , cv2.CV_64F,  np.array(sobel_x))
		dy= cv2.filter2D(img , cv2.CV_64F,  np.array(sobel_y))
		img=exposure.equalize_hist(dx+dy)

		cv2.imwrite(out_filename, ScaleImage(img,0,255).astype(np.uint8))


"""
!ffmpeg \
  -framerate 60  \
  -pattern_type glob \
  -i '/mnt/c/data/visus-datasets/Pania_2021Q3_in_situ_data/EFRC_Related_Data_For_External_Collaboration/animation_sobel/*.tiff' \
  -vf scale 1080:1280  \
  -c:v libx264 out.mp4
"""