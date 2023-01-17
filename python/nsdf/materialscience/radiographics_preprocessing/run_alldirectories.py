import sys
from preprocessRadiographs import *
from pathlib import Path

if __name__ == '__main__':


    radiographics_dir=sys.argv[1]
    directory_preprocessed = sys.argv[2]
    directory_averaged = sys.argv[3]
    for dir in os.listdir(radiographics_dir):
        if dir.startswith("radiographic_scan_id"):
            print("Processing "+dir)
            radio_dir=os.listdir(radiographics_dir+dir+"/")
            if len(radio_dir)==0:
                print("No files in dir:" + str(len(radio_dir)))
                pass
            else:
                directory_preprocess = dir
                Path(directory_preprocessed+"/"+dir).mkdir(parents=True, exist_ok=True)
                Path(directory_averaged+"/"+dir).mkdir(parents=True, exist_ok=True)

                global_directories(radiographics_dir+dir+"/", directory_preprocessed+dir+"/", directory_averaged+dir+"/")
                # Pre-processing the radiographs in parallel per set
                pool_obj = multiprocessing.Pool()
                original_images=radio_dir
                pool_obj.map(preprocess,original_images)
                print("Finish preprocessing")

                pool_obj = multiprocessing.Pool()
                # Averaging by 4 the pre-processed radiographs
                images=os.listdir(directory_preprocessed+dir+"/")
                n_averages=4
                images.sort()
                sets=int(len(images)/n_averages)
                four_split=np.array_split(images,sets)
                pool_obj.map(average,four_split)
                print("Finish averaging")



