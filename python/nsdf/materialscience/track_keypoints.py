import cv2 
import sys,io,types,threading,random,glob,math
import os,sys,time
import numpy as np
import tifffile
from skimage import exposure

import bokeh.io
import bokeh.events
import bokeh.models
import bokeh.plotting 

# //////////////////////////////////////////////////////////////////////////
def IntPoint(p):
	return (int(p[0]),int(p[1]))

# ////////////////////////////////////////////////////////////////////////////////////
class Canvas:
  
	# constructor
	def __init__(self):
		self.color_mapper = bokeh.models.LinearColorMapper(palette="Greys256")
		self.color_mapper.low=0
		self.color_mapper.high=1.0
		self.color_bar = bokeh.models.ColorBar(color_mapper=self.color_mapper)  
		self.figure=bokeh.plotting.Figure(active_scroll = "wheel_zoom")
		self.figure.x_range = bokeh.models.Range1d(0,1024)   
		self.figure.y_range = bokeh.models.Range1d(0, 768) 
		self.figure.toolbar_location="below"
		self.figure.sizing_mode = 'stretch_both'
		self.source_image = bokeh.models.ColumnDataSource(data={"image": [np.zeros((768,1024),dtype=np.float64)], "x":[0], "y":[0], "dw":[1024], "dh":[768]})  
		self.figure.image("image", source=self.source_image, x="x", y="y", dw="dw", dh="dh", color_mapper=self.color_mapper)
		self.figure.add_layout(self.color_bar, 'right')
		self.status = {}
		self.callback = bokeh.io.curdoc().add_periodic_callback(self.onTimer, 100) 
  
		self.images=[]
		self.keypoints=[]
		self.scatters=[]
		self.colors=[]
  
		self.timestep = bokeh.models.Slider(title='Time', value=0, start=0, end=1, sizing_mode="stretch_width")
		self.timestep.on_change ("value",lambda attr, old, new: self.setTimestep(int(new)))  
  
		self.layout=bokeh.layouts.column(
    	bokeh.layouts.row(self.timestep,sizing_mode='stretch_width'),
      self.figure,
     	sizing_mode='stretch_both')

	# refresh
	def refresh(self):
		pass

	# onTimer
	def onTimer(self):
		# ready for jobs?
		canvas_w,canvas_h=(self.getWidth(),self.getHeight())
		if canvas_w==0 or canvas_h==0 :
			return
 
		# simulate fixAspectRatio (i cannot find a bokeh metod to watch for resize event)
		if self.status.get("w",0)!=canvas_w or self.status.get("h",0)!=canvas_h:
			self.setViewport(*self.getViewport())
			self.status["w"]=canvas_w
			self.status["h"]=canvas_h
			self.refresh()

	# getWidth
	def getWidth(self):
		return self.figure.inner_width

	# getHeight
	def getHeight(self):
		return self.figure.inner_height

  	# getViewport
	def getViewport(self):
		x1,x2=self.figure.x_range.start, self.figure.x_range.end
		y1,y2=self.figure.y_range.start, self.figure.y_range.end 
		return (x1,y1,x2,y2)

  	# getViewport
	def setViewport(self,x1,y1,x2,y2):
		# print("setViewport",x1,y1,x2,y2)
		# fix aspect ratio
		W,H=self.getWidth(),self.getHeight()
		if W and H: 
			ratio=W/H
			w, h, cx, cy=(x2-x1),(y2-y1),0.5*(x1+x2),0.5*(y1+y2)
			w,h=(h*ratio,h) if W>H else (w,w/ratio) 
			x1,y1,x2,y2=cx-w/2,cy-h/2, cx+w/2, cy+h/2
		self.figure.x_range.start=x1
		self.figure.x_range.end  =x2
		self.figure.y_range.start=y1
		self.figure.y_range.end  =y2

	# addImage
	def addImage(self,img, keypoints):
		print("addImage","img.shape",img.shape, "img.dtype", img.dtype,"#keypoints",len(keypoints))
		# is the first image?
		m,M=np.min(img),np.max(img)
		self.images.append(img)
		self.keypoints.append(keypoints)
		self.timestep.end=len(self.images)-1
  
		if len(self.images)==1:
			self.setViewport(0,0,img.shape[0],img.shape[1])
			self.color_mapper.low =m
			self.color_mapper.high=M
		else:
			self.color_mapper.low =min(m,self.color_mapper.low )
			self.color_mapper.high=max(M,self.color_mapper.high)
 
			self.setTimestep(0)

	# getTimestep
	def getTimestep(self):
		return self.timestep.value

	# setTimestep
	def setTimestep(self,value):
		value=max(0,min(value,len(self.images)-1))
		self.timestep.value=value
		img=self.images[value]
		self.source_image.data={"image":[img], "x":[0], "y":[0], "dw":[img.shape[1]], "dh":[img.shape[0]]}

		# draw keypoints
		while self.scatters:
			self.figure.renderers.remove(self.scatters[-1])
			self.scatters.pop()
		x1=[k.pt[0] for k in self.keypoints[value]]
		y1=[k.pt[1] for k in self.keypoints[value]]
		r1=[k.size  for k in self.keypoints[value]]
		while len(self.colors)<len(x1):
			self.colors.append((random.randint(0,255),random.randint(0,255),random.randint(0,255)))
		self.scatters.append(self.figure.scatter(x=x1, y=y1, radius=r1, color=self.colors, alpha=0.2, marker="circle") )

		# render the path
		xs,ys=[],[]
		for keypoint_t in zip(*self.keypoints):
			xs.append([k.pt[0] for k in keypoint_t])
			ys.append([k.pt[1] for k in keypoint_t])
		self.scatters.append(self.figure.multi_line(xs=xs,ys=ys,line_color =self.colors,alpha=1.0,line_width =3))
      
  
	# goPrev
	def goPrev(self):
		self.setTimestep(value-1)

	# goNext
	def goNext(self):
		self.setTimestep(value+1)


# //////////////////////////////////////////////////////////////////////////
def Denoise(img, sigma):
	assert img.dtype==np.float64
	blur=cv2.GaussianBlur(img,ksize=(0,0), sigmaX=sigma,	sigmaY=sigma)
	ret=cv2.subtract(img,blur) 
	print("Denoise",ret.shape,ret.dtype,np.min(ret),np.max(ret))
	return ret

# //////////////////////////////////////////////////////////////////////////
def SetRange(img,A,B):
	m,M=np.min(img),np.max(img)
	ret=(A + B*(img-m)/(M-m))
	print("SetRange",ret.shape,ret.dtype,np.min(ret),np.max(ret))
	return ret

# //////////////////////////////////////////////////////////////////////////
def FilterKeypoints(keypoints,descriptors, indices):
	print("FilterKeypoints",len(indices))
	return (
  	[keypoints[it] for it in indices],
		np.asarray([descriptors[it] for it in indices]))

# //////////////////////////////////////////////////////////////////////////
def FindKeypoints(detector, img, mask=None):
	assert img.dtype==np.uint8 # swift works only on 
	t1=time.time()
	keypoints, descriptors =detector.detectAndCompute(img,mask)
	print("Found",len(keypoints),"keypoints in",time.time()-t1,"seconds")
	assert all([k.size>=0 for k in keypoints])
	return keypoints,descriptors
	
# //////////////////////////////////////////////////////////////////////////
def FindMathes(d1,d2,algorithm="bf",ratio_check=0.8):
	t1=time.time()
 
	if algorithm=="bf":
			matcher=cv2.BFMatcher()
			matches = matcher.knnMatch(d1, d2, k=2) if ratio_check else matcher.match(d1, d2)
			
	elif algorithm=="flann":
		flann=cv2.FlannBasedMatcher(dict(algorithm = 0, trees = 5),dict(checks=50))
		matches = flann.knnMatch(d1, d2, k=2) if ratio_check else flann.match(d1, d2)
   
	if ratio_check:
		matches = [m for m, n in matches if (m.distance / float(n.distance)) < 0.8]

	print(f"FindMathes {algorithm}",len(matches),"in",time.time()-t1,"seconds") 
  
	return matches


# ////////////////////////////////////////////////////////////////////////////////
"""_summary_
python3 -m bokeh serve ./python/keypoints.py  --dev
"""

if "bokeh" in __name__:

	pattern="/mnt/c/data/visus-datasets/Pania_2021Q3_in_situ_data/EFRC_Related_Data_For_External_Collaboration/animation_bob/*.tiff"
	pattern="/mnt/c/data/visus-datasets/Pania_2021Q3_in_situ_data/EFRC_Related_Data_For_External_Collaboration/animation_sobel/*.tiff"
	filenames=sorted(glob.glob(pattern))
	filenames=[filenames[0],filenames[200]]
	print("Found",len(filenames),"filenames")
	canvas=Canvas() 
 
	detector=cv2.SIFT_create(
			nfeatures = 0,               # default 0     The number of best features to retain. The features are ranked by their scores (measured in SIFT algorithm as the local contrast)
			nOctaveLayers = 3,           # default 3     The number of layers in each octave. 3 is the value used in D. Lowe paper. The number of octaves is computed automatically from the image resolution.
			contrastThreshold =0.04*0.1, # default 0.04  The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. The larger the threshold, the less features are produced by the detector.
			edgeThreshold = 10*10,       # defaul 10     The threshold used to filter out edge-like features. Note that the its meaning is different from the contrastThreshold, i.e. the larger the edgeThreshold,the less features are filtered out (more features are retained).
			sigma = 1.6                  # default 1.6. # The sigma of the Gaussian applied to the input image at the octave #0. If your image is captured with a weak camera with soft lenses, you might want to reduce the number.
		) 
 
	# good in the internal
	# detector=cv2.AKAZE.create(threshold=0.0001)
	# detector=cv2.ORB.create()

	# OPEN QUESTIONS
	# 1. better to keypoint detection on the original image or with noise removal
	# 2. is SIFT robust to range changes?

	def MyJob(filename):
		img=tifffile.imread(filename)
		print("Loaded image",filename, img.shape,img.dtype)
		keypoints,descriptors=FindKeypoints(detector, img)
		return img,keypoints,descriptors  
 
	from multiprocessing.pool import ThreadPool
	results=list(zip(*ThreadPool(32).map(MyJob, filenames)))
	IMAGES,KEYPOINTS,DESCRIPTORS=[list(it) for it in results]
	
	# find matches (0...J-1 J)
	for J in range(1,len(IMAGES)):
		k1,d1=KEYPOINTS[J-1],DESCRIPTORS[J-1]
		k2,d2=KEYPOINTS[J+0],DESCRIPTORS[J+0]
		t1=time.time()
		print("Finding matches...")
		matches=FindMathes(d1,d2,ratio_check=0.8)
		print("Found",len(matches),"in",time.time()-t1,"seconds")
 
		# march should not be too far 
		if True:
			MAX_DISTANCE=2160*0.05 # 5% == 108
			good=[]
			for match in matches:
				x1,y1=k1[match.queryIdx].pt
				x2,y2=k2[match.trainIdx].pt
				#if (x2-x1)<0: continue
				if ((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))>MAX_DISTANCE*MAX_DISTANCE: continue
				#if math.fabs(y2-y1)>math.fabs(x2-x1): continue
				good.append(match)
			print("Filtering by max distance",len(matches),"->",len(good))
			matches=good
 
		# all the previous
		for I in range(0,J):
			KEYPOINTS[I],DESCRIPTORS[I]=FilterKeypoints(KEYPOINTS[I],DESCRIPTORS[I],[match.queryIdx for match in matches])

		if True: 
			KEYPOINTS[J],DESCRIPTORS[J]=FilterKeypoints(KEYPOINTS[J],DESCRIPTORS[J],[match.trainIdx for match in matches])
  
	for I in range(len(IMAGES)):
		canvas.addImage(IMAGES[I], KEYPOINTS[I])
 
	bokeh.io.curdoc().add_root(canvas.layout)

