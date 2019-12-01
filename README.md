# Vertebral body and PMMA segmentation using 3dunet

This is final project source code of CS470 in 2019 fall semester. We used 3d u-net to get vertebral body and PMMA segmentation results from MRI images.  

## Prerequisites
- Linux
- NVIDIA GPU
- CUDA CuDNN

## Getting Started

### Dependencies
- pytorch (0.4.1+)
- torchvision (0.2.1+)
- tensorboardx (1.6+)
- h5py
- scipy 
- scikit-image
- scikit-learn (0.21+)
- pytest
- hdbscan (0.8.22+)
- pynrrd
- OpenCV-Python

Setup a new conda environment with the required dependencies via:
```
conda create -n 3dunet pytorch torchvision tensorboardx h5py scipy scikit-image scikit-learn pyyaml hdbscan pytest -c conda-forge -c pytorch
``` 
Activate newly created conda environment via:
```
source activate 3dunet
```
## Data preprocessing
You cannot run the data preprocessing code because there is no dataset. We used following command to preprocess the model.
```
python nrrd_to_HDF5.py
```

## Train
You cannot train the model because there is no dataset. We used following command to train the model.
```
python train.py --config resources/train_config_ce_191201v2.yaml
```

## Test
If you want to run the model with test data, type the following code in the command. 
```
python predict.py --config resources/test_config_ce_191201.yaml
```
Prediction will be saved to `CS470/result/191201_final`.

## Visualization 
You can generate videos and calculate accuracy using data produced by 'predict.py'. You need to install nrrd and opencv. Type the following commands. 
```
pip install pynrrd
pip install OpenCV-Python
```
If this error occurs, 'ImportError: libgthread-2.0.so.0: cannot open shared object file: No such file or directory', type the following commands.
```
apt-get update
apt-get install libgtk2.0-dev
```
If the environment is ready, type the following code. 
```
python CS470/HDF5_visualize.py
```
Video and accuracy will be saved to `CS470\result\191201_final_vis`.


If you want to convert HDF5 file to nrrd file, type the following code. You can see the result with MITK (http://mitk.org/wiki/MITK) and [MITK_plugin](MITK_plugin.zip)
```
python CS470/HDF5_to_nrrd.py
```