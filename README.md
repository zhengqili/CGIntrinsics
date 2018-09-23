# CGIntrinsics

This is the CGIntrinsics implementation described in the paper "CGIntrinsics: Better Intrinsic Image Decomposition through Physically-Based Rendering, Z. Li and N. Snavely, ECCV 2018" (Still Updating, please stay tune). The code skeleton is based on "https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix" and "https://github.com/lixx2938/unsupervised-learning-intrinsic-images". If you use our code for academic purposes, please consider citing:

    @inproceedings{li2018cgintrinsics,
	  	title={CGIntrinsics: Better Intrinsic Image Decomposition through Physically-Based Rendering},
	  	author={Zhengqi Li and Noah Snavely},
	  	booktitle={European Conference on Computer Vision (ECCV)},
	  	year={2018}
	}
  

#### Dependencies & Compilation:
* The code was written in Pytorch 0.2 and Python 2, but it should be easy to adapt it to Python 3 version and Pytorch 0.3/0.4 if needed. 


#### Training on the CGIntrinsics dataset:
* Download the CGIntrinsics dataset (intrinsics_final) from our website: 
* Download IIW densely connected pair-wise juedgement we precomputed in our website and original images and datta in original website. 
* Download SAW list in our website and original data in original SAW website.
* create a directory in your machine called CGIntrinsics
* In CGIntrinsics, you should have 3 folders (1) intrinsics_final (2) IIW (3) SAW. You should put original IIW png and json you download from original website in CGIntrinsics/data/, and you should put 

* In train.py file, you should root and and full_root variable to your root directory for putting CGIntrinsics folder.
* In options/base_options.py, you should change --gpu_ids to fit your machine GPUs.
* run:
```bash
    python train.py
```
