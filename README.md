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

  ```
  DATASETS = {
	'CUHK': {
		'transforms': trans_conf.EncodeTransforms,
		'train_source_root': dataset_paths['CUHK_train_P'],
		'train_target_root': dataset_paths['CUHK_train_S'],
		'test_source_root': dataset_paths['CUHK_test_P'],
		'test_target_root': dataset_paths['CUHK_test_S'],
	},}
  ```



### Training and Inference
* Train a model

```
CUDA_VISIBLE_DEVICES="0" python scripts/train.py \
--dataset_type=CUHK \
--encoder_type=hifinet \
--exp=experiments/CUHK \
--workers=1 \
--batch_size=4 \
--test_batch_size=2 \
--test_workers=1 \
--val_interval=5000 \
--save_interval=5000 \
--n_iters_per_batch=1 \
--max_val_batches=150 \
--output_size=1024 \
--load_w_encoder
```

* inference the model

```
CUDA_VISIBLE_DEVICES="0" python scripts/inference.py \
--exp=experiments/CUHK \
--checkpoint_path=/model/path \
--data_path=/your/test/data/path \
--test_batch_size=4 \
--test_workers=4 \
--n_iters_per_batch=2 \
--load_w_encoder \
--w_encoder_checkpoint_path pretrained_models/faces_encoder.pt 
```
and you can use '--save_weight_deltas' to save the final weight.

### Editing
# You can edit the generated image through text by：

```
python editing/edit/edit.py \
--exp /your/experiment/dir \
--weight_deltas_path /your/weight_deltas \
--neutral_text "a face" \
--target_tex "a face with glasses"
```



**bibtex:**

```latex
@article{peng2023hifisketch,
  title={HiFiSketch: High Fidelity Face Photo-Sketch Synthesis and Manipulation},
  author={Peng, Chunlei and Zhang, Congyu and Liu, Decheng and Wang, Nannan and Gao, Xinbo},
  journal={IEEE Transactions on Image Processing},
  year={2023},
  publisher={IEEE}
}
```

## Acknowledgments

Our code is inspired by [Hyperstyle](https://github.com/yuval-alaluf/hyperstyle) and [stylegan2](https://github.com/yuval-alaluf/hyperstyle)
