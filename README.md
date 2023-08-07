# WBC classification and segmentation
 This repository contains codes for classification and segmentation of white blood cells.

# requirements.txt file may be used to create an environment using:
# $ conda create --name <env> --file <this file>

Extract the dataset blood_cell_images.zip and keep the python files and the extracted "blood_cell_images" folder in the same working directory

Run induvidual cells of the following python files sequentially in a python editor.
	mia_project.py --> for classification models
	mia_seg.py --> for segmentation methods
	mia_grad_cam.py (experimental - ran into GPU memory allocation issues. May run on bigger GPUs) 
			--> Grad-CAM implementation

See the commented lines in the cells for instructions and summaries of what each cell basically does.
