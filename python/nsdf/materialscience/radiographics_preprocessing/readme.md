## Preprocessing radiographic for strain measurement
This workflow preprocess radiographic data to measure the amount of stretch and compression along Silica or any other material. There are two main functions: preprocess and average.  
Steps in preprocess:
 * Load the slices in the image stack
 * Gaussian blurred with sigma=30
 * Subtract the blurred image from the original image
 * Adjust brightness and contrast
 * Rotate the image 90 degrees clockwise  
In average, we take by _n_ time consecutive images and average them to reduce the noise.  
### Installation
The software stack required to run the workflow can be installed using Anaconda. Each of these options will installed the required dependencies.  
**Dependencies:**
  * python=3.9
  * numpy
  * pillow
  * pip
  * opencv=4.5
  * qt
  * pyqt
  * pip:
  * scikit-image
  * panel
  * vtk
  * OpenVisus
  * tifffile

If you do not have Anaconda installed, you can follow the instructions here to install it. Make sure to change the prefix in `env_conda` to the location of Anaconda in your local machine (e.g., `/opt/anaconda3/`, `/home/opt/anaconda3/`)

Run the next commands on your local machine:
```
conda env create -f env_conda.yaml
conda activate materials-science
```

### How it works?
There are *N* radiographic sets, and each set includes different tiff images that need to be processed and average within the set. We run one set per node (data parallelism) and each node processes the images in parallel using all available cores (multithreading). Currently, there are 95 sets from 8 pillars of Silica. Each set contains from 25 to 900 images.  
To run the workflow, you use `preprocess_radiographs.py` and provide the directory with the original radiographs (`directory_preprocess`), the directory where you will store the preprocessed images (`directory_preprocessed`), and the directory where the averages are stored (`directory_averaged`). 
For example:
```
$ python3 preprocess_radiographs.py radigraphics/radiographic_scan_id_112536 preprocessed/radiographic_scan_id_112536 averaged/radiographic_scan_id_112536 
```
To run all datasets, we use `run_alldirectories.py` where you need to provide the directory where all original radiographics are stored.
```
$ python3 run_alldirectories.py radigraphics/ preprocessed averaged 
```
We have also run this workflow using Kubernetes. The instructions on how to run are in the `ks` directory. 
