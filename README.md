# HiFiSketch

  We provide `PyTorch` implementations for our TIP2023 paper [`HifiSketch`]([https://ieeexplore.ieee.org/abstract/document/9845477]): 

```latex
@article{peng2023hifisketch,
  title={HiFiSketch: High Fidelity Face Photo-Sketch Synthesis and Manipulation},
  author={Peng, Chunlei and Zhang, Congyu and Liu, Decheng and Wang, Nannan and Gao, Xinbo},
  journal={IEEE Transactions on Image Processing},
  year={2023},
  publisher={IEEE}
}
```

This project can generate face sketch from photos and edit the composite portrait using text.
[`Paper@IEEE`]([https://ieeexplore.ieee.org/abstract/document/9845477])   [`Code@Github`]([https://github.com/shenhaiyoualn/EADT])  



## Prerequisites

- Linux 
- Python 3.7
- NVIDIA GPU + CUDA + CuDNN

## Getting Started

### Installation

* Clone this repo: 

  ```
  git clone https://github.com/shenhaiyoualn/HiFiSketch
  cd HiFiSketch
  ```

*The environment file is defined in environment/environment.yaml, Install all the dependencies by:
  ```
conda env create -f environment.yml
  ```

### Prepare
* Download your dataset and put it in the dataset directory, then update configs/path_config.py like:

  ```
  dataset_paths = {
    'CUHK_train_P': '/media/gpu/T7/HifiSketch/datasets/CUHK_train_Photo',
    'CUHK_train_S': '/media/gpu/T7/HifiSketch/datasets/CUHK_train_Sketch',
	  'CUHK_test_P': '/media/gpu/T7/HifiSketch/datasets/CUHK_test_Photo',
    'CUHK_test_S': '/media/gpu/T7/HifiSketch/datasets/CUHK_test_Sketch',}
  ```
* Update configs/data_conf.py like:
  DATASETS = {

  ```
DATASETS = {
	'CUHK': {
		'transforms': trans_conf.EncodeTransforms,
		'train_source_root': dataset_paths['CUHK_train_P'],
		'train_target_root': dataset_paths['CUHK_train_S'],
		'test_source_root': dataset_paths['CUHK_test_P'],
		'test_target_root': dataset_paths['CUHK_test_S'],
	},
 }
  ```



### train/test
* Train a model

  ```
  python train.py --dataset_train_list train_sample.txt --dataset_test_list test_sample.txt   --name eadt
  ```

* Test the model

  ```
  python test.py  --input_size 256  --checkpoint_dir /home/sd01/EADT/checkpoint/eadt.ckpt
  ```


### Preprocessing steps

If you need to use your own data, please align all faces by eyes and the face parsing is segmented by [face-parsing](https://github.com/jehovahxu/face-parsing.PyTorch)


## Citation

 If you use this code for your research, please cite our paper. 

> Zhang, C., Liu, D., Peng, C., Wang, N., & Gao, X. (2022). Edge Aware Domain Transformation for Face Sketch Synthesis. IEEE Transactions on Information Forensics and Security, 17, 2761-2770. (Accepted)

**bibtex:**

```latex
@article{zhang2022edge,
  title={Edge Aware Domain Transformation for Face Sketch Synthesis},
  author={Zhang, Congyu and Liu, Decheng and Peng, Chunlei and Wang, Nannan and Gao, Xinbo},
  journal={IEEE Transactions on Information Forensics and Security},
  volume={17},
  pages={2761--2770},
  year={2022},
  publisher={IEEE}
}
```

## Acknowledgments

Our code is inspired by [GENRE](https://github.com/fei-hdu/genre) and [SPADE/GauGAN](https://github.com/NVlabs/SPADE).
