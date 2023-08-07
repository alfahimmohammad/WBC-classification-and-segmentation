# WBC classification and segmentation
 This repository contains codes for the classification and segmentation of white blood cells. The DL models are compiled and trained using Keras and TensorFlow.

requirements.txt file may be used to create an environment using the following:

`$ conda create --name <env> --file requirements.txt`

Extract the dataset blood_cell_images.zip and keep the Python files and the extracted "blood_cell_images" folder in the same working directory.

- Run individual cells of the following Python files sequentially in a Python editor.
  - mia_project.py --> for classification models
  - mia_seg.py --> for segmentation methods
  - mia_grad_cam.py (experimental - ran into GPU memory allocation issues. May run on bigger GPUs) --> Grad-CAM implementation

Please look at the comments in the cells for instructions and summaries of what each cell does.

Model checkpoints can be downloaded from [here](https://drive.google.com/drive/folders/1ufI-6CeJj8Qkyle-jrVBGW5Umm1bRJJT?usp=sharing)
