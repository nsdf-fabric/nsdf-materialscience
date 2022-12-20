import os,sys,glob
import cv2 as cv
import numpy as np
import multiprocessing
import tifffile

# /////////////////////////////////////////////////////////////////////////////////
def BobPreprocess(filename):
	img = cv.imread(filename, cv.IMREAD_GRAYSCALE);assert img.dtype==np.uint8
	img=img.astype(np.float32)
	img = cv.subtract(img,cv.GaussianBlur(img,(0,0), 30.0))
	img = cv.addWeighted(img,25,img,0,125)
	img=cv.convertScaleAbs(img, alpha=2.0, beta=-125)
	img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
	return img.astype(np.uint8)

# /////////////////////////////////////////////////////////////////////////////////
def BobEquivalent(filename):
	img=tifffile.imread(filename);assert img.dtype==np.uint16
	img = img * (255.0/(2**16));assert img.dtype==np.float64
	img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
	img = img - cv.GaussianBlur(img,(0,0), 30.0)
	img = 25.0*img+125
	img=np.abs(2.0*img-125.0)
	img[img>255.0]=255.0
	return img.astype(np.uint8)  

# /////////////////////////////////////////////////////////////////////////////////
def BobAverage(images):
	return np.sum([(1.0/len(images))*img for img in images])

# ////////////////////////////////////////////////////////////////////////
def SaveUint8Image(filename,img):
	cv.imwrite(filename, img)

# ////////////////////////////////////////////////////////////////////////
if __name__ == '__main__':

	SRC="/mnt/c/data/visus-datasets/Pania_2021Q3_in_situ_data/EFRC_Related_Data_For_External_Collaboration/Radiographic_Scans_Tiffs"
	DST="/mnt/c/data/visus-datasets/Pania_2021Q3_in_situ_data/EFRC_Related_Data_For_External_Collaboration/animation_bob"

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
		for in_filename in sorted(glob.glob(f"{SRC}/{seq}/*.tiff")):
			out_filename=f"{DST}/{seq}_{os.path.basename(in_filename)}"
			print("Doing",in_filename,out_filename)
			img=BobPreprocess(in_filename)
			SaveUint8Image(out_filename, img)

"""
ffmpeg \
  -framerate 60  \
  -pattern_type glob -i '/mnt/c/data/visus-datasets/Pania_2021Q3_in_situ_data/EFRC_Related_Data_For_External_Collaboration/animation/*.tiff' \
  -vf scale 1080:1280  \
  -c:v libx264 out.mp4

"""