import os
import sys
import cv2 as cv
import numpy as np
import multiprocessing
from pathlib import Path

#This script preprocessed rasiographs to measure the amount of stretch and compression along Silica or any other material

def global_directories(name_dir, name_preprocessed_dir, name_average_dir):
    global directory
    global preprocessed_directory
    global average_directory
    directory=name_dir #from where you are reading the images (directory)
    preprocessed_directory = name_preprocessed_dir #where you are writing the pre-processed images (preprocessed_directory)
    average_directory = name_average_dir

# This function preprocess each image from the radiograph set with a series of steps 
def preprocess(image):
    # Read original image
    print("Reading image: ", image)
    original_img = cv.imread(directory+"/"+image, cv.IMREAD_GRAYSCALE).astype(np.float32)

    # Gaussian blur with sigma=30
    sigma = 30.0
    blurred_img = cv.GaussianBlur(original_img,(0,0), sigma)

    # Substract blurred image from original image
    substracted_img = cv.subtract(original_img,blurred_img)

    # Adjust contrast and brightness
    alpha = 25
    beta = 125
    #https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
    enhanced_img = cv.addWeighted(substracted_img,alpha,substracted_img,0,beta)

    alpha= 2.0
    beta=-125
    #https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
    contrast_img=cv.convertScaleAbs(enhanced_img, alpha=alpha, beta=beta)

    # Rotate image 90 degrees clockwise 
    rotated_img = cv.rotate(contrast_img, cv.ROTATE_90_CLOCKWISE)

    # Save results
    cv.imwrite(preprocessed_directory+"/Preprocessed_"+image, rotated_img.astype(np.uint8))

# This function averaged by N=4 preprocessed images 
def average(images):
    # Get index from first and last image
    names="Radiographs_"+str(images[0].split("__")[1].split(".")[0])+"--"+str(images[-1].split("__")[1].split(".")[0])
    #print("Read: ", preprocessed_directory+images[0])
    # Read first image and make it equal to avg_img
    avg_image = cv.imread(preprocessed_directory+images[0], cv.IMREAD_GRAYSCALE)
    #print(len(images))
    # Function taken from: https://leslietj.github.io/2020/06/28/How-to-Average-Images-Using-OpenCV/
    for i in range(len(images)):
        if i == 0:
            pass
        else:
            # Read next image
            image=cv.imread(preprocessed_directory+"/"+images[i], cv.IMREAD_GRAYSCALE)
            alpha=1.0/(i+1)
            beta=1.0-alpha
            # Average image
            avg_image = cv.addWeighted(image, alpha, avg_image, beta, 0.0)
    cv.imwrite(average_directory+"/Average_"+names+".tiff", avg_image)


if __name__ == '__main__':

    directory_preprocess = sys.argv[1]
    directory_preprocessed = sys.argv[2]
    directory_averaged = sys.argv[3]

    Path(directory_preprocessed).mkdir(parents=True, exist_ok=True)
    Path(directory_averaged).mkdir(parents=True, exist_ok=True)
    global_directories(directory_preprocess, directory_preprocessed, directory_averaged)
    # Pre-processing the radiographs in parallel per set 
    pool_obj = multiprocessing.Pool() 
    original_images=os.listdir(directory_preprocess)
    pool_obj.map(preprocess,original_images)
    print("Finish preprocessing")

    pool_obj = multiprocessing.Pool() 
    # Averaging by 4 the pre-processed radiographs 
    images=os.listdir(directory_preprocessed)
    n_averages=4
    images.sort()
    sets=int(len(images)/n_averages)
    four_split=np.array_split(images,sets)
    pool_obj.map(average,four_split)
    print("Finish averaging")

