# SOGAN

This is the implementation code for "Spatial Orthogonal Attention Generative Adversarial Network for MRI Reconstruction".

The code was written by Wenzhong Zhou.

If you use this code for your research, please cite our paper.

# Prerequisites

The original code is in python 3.5 under the following dependencies:
1. tensorflow (v1.8.0)
2. tensorlayer (v1.11.0)
3. easydict 
4. nibabel
5. scikit-image 

Code tested in Ubuntu 16.04 with Nvidia GPU + CUDA CuDNN.

# How to use the code

1. Prepare data

    1) Data used in this work are publicly available from the MICCAI 2013 grand challenge ([link](https://my.vanderbilt.edu/masi/workshops/)). We refer users to register with the grand challenge organisers to be able to download the data.
    2) Download training and testing data respectively into data/Dataset/Training and data/Dataset/Testing. (We randomly included 100 T1-weighted MRI datasets for training and 50 datasets for testing)
    3) run 'python3 data_loader.py'
    4) after running the code, training/validation/testing data should be saved to 'data/Dataset/' in pickle format.
    
2. Train model

   run 'python3 train.py'

# Results

Please refer to the paper for detailed results.

# Thanks to the Third Party Libs
[DAGAN](https://github.com/nebulaV/DAGAN) 




