# Multi-Camera People Tracking

This is project is the implementation of the official repositroy for 7th NVIDIA AI City Challenge (2023) [Track 1: Multi-Camera People Tracking](https://github.com/ipl-uw/AIC23_Track1_UWIPL_ETRI) [[Arxiv]](https://arxiv.org/abs/2304.09471)
 
## Overall Pipeline

<img src="figure.jpg" width="650" />

## Directory Structure

1. Make sure to use ```uncc-kubeflow-base``` docker image 
2. The complete project is available at ``` path ```
3. Download the dataset from https://www.aicitychallenge.org/, and place the test data in `./data/`

You should see the `data` folder organized as follows: 
```
data
├── annotations
│   ├── fine_tune
│   │   ├── train_hospital_val_hospital_sr_20_0_img_15197.json
│   │   ├── train_market_val_market_sr_20_0_img_19965.json
│   │   ├── train_office_val_office_sr_20_0_img_20696.json
│   │   └── train_storage_val_storage_sr_20_0_img_15846.json
│   └── train_all_val_all_sr_20_10_img_77154.json
├── train
│   ├── S002
│   │   ├── c008
│   │   │   ├── frame
│   │   │   ├── label.txt
│   │   │   └── video.mp4
│   .   .
│   .   .
├── validation
├── test
│   ├── S002
│   │   ├── c008
│   │   │   ├── video.mp4
```


## Enviroment Requirements

 ```
    git clone https://github.com/ipl-uw/AIC23_Track1_UWIPL_ETRI.git
    cd AIC23_Track1_UWIPL_ETRI
 ```
The implementation of our work is built upon [BoT-SORT](https://github.com/NirAharon/BoT-SORT), [OpenMMLab](https://github.com/open-mmlab), and [torchreid](https://github.com/KaiyangZhou/deep-person-reid). We also adapt [Cal_PnP](https://github.com/zhengthomastang/Cal_PnP) for camera self-calibration.

Four different enviroments are required for the reproduce process. Please install these three enviroments according to the following repos:
1. [Installation for mmyolo*](https://github.com/open-mmlab/mmyolo#%EF%B8%8F-installation-)
2. [Installation for mmpose*](https://mmpose.readthedocs.io/en/latest/installation.html)
3. [Installation for torchreid*](https://github.com/KaiyangZhou/deep-person-reid#installation)
4. [Installation for BoT-SORT](https://github.com/NirAharon/BoT-SORT#installation)

\* optional for fast reproduce


### Bot-SORT Installtion
```
#Bot-SORT
conda create -n botsort_env python=3.7
conda activate botsort_env

# pytorch
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 

cd BoT-SORT
pip3 install -r requirements.txt
python3 setup.py develop
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Cython-bbox
pip3 install cython_bbox

# faiss cpu / gpu
pip3 install faiss-cpu
pip3 install faiss-gpu
```

#### mmcv Installation for mmyolo, mmpose
<!--
```
### mmyolo Installtion
Make sure to review Issues section before mmyolo installation.

# train-detector
pip install future tensorboard
pip install setuptools==59.5.0
```
-->
## Inferencing

#### Get Detection (skip for fast reproduce)
0. To Fast Reproduce

Directly use the txt files in the `data/test_det` folder and skip the following steps.

1. Prepare Models

- Download the pretrained YOLOX_x from [ByteTrack [Google Drive]](https://drive.google.com/file/d/1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5/view)
- Download (or train from scratch) the YOLOv7 weights from [[Google Drive]](https://drive.google.com/drive/folders/10LT1BlBAfYnr-fJjka_Lnzf4nH0N723-?usp=share_link)

2. Get Real (S001) detection
```
bash scripts/3_inference_det_real.sh
```

3. Get Synthetic detection
```
bash scripts/4_inference_det_syn.sh
```

#### Get Embedding 
0. To Fast Reproduce
Download the [embedding npy files](https://drive.google.com/drive/folders/1qbwu37PlFSxmJIBLzJq1L9cAwryahAj7) and put all the npy files under `data/test_emb`, then you can skip step 1 and 2.

1. Prepare Models (optional)
* Download the [ReID model](https://drive.google.com/file/d/1cP-3esZSnktw64SXMHn5cDk6BX15e2-q/view?usp=sharing) for synthetic dataset
* Download the pretrained ReID models from [torchreid](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO). Including [osnet_ms_m_c](https://drive.google.com/file/d/1UxUI4NsE108UCvcy3O1Ufe73nIVPKCiu/view), [osnet_ibn_ms_m_c](https://drive.google.com/file/d/1Sk-2SSwKAF8n1Z4p_Lm_pl0E6v2WlIBn/view), [osnet_ain_ms_m_c](https://drive.google.com/file/d/1YjJ1ZprCmaKG6MH2P9nScB9FL_Utf9t1/view), [osnet_x1_0_market](https://drive.google.com/file/d/1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA/view), [osnet_x1_0_msmt17](https://drive.google.com/file/d/112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M/view)
* Put all the models in deep-person-reid/checkpoints

2. Get Appearance Embedding (optional)
```
bash scripts/5_inference_emb.sh
```

#### Run Tracking

The root_path for the following command should set to the repo's loaction

1. Navigate to the BoT-SORT folder
```
cd BoT-SORT
```

2. Run tracking - (Make sure to have Sufficient RAM(atleast 35 gigs) before executing run_tracking.py)
```
conda activate botsort_env
python tools/run_tracking.py <root_path>
```


4. Conduct spatio-temporal consistency reassignment 
```
python STCRA/run_stcra.py <input_tracking_file_folder> <output_tracking_file_folder>
```
<!--
5. Generate final submission
```
cd ../BoT-SORT
python tools/aic_interpolation.py <root_path>
python tools/boundaryrect_removal.py <root_path>
python tools/generate_submission.py <root_path>
```
-->

## Issues and Resolutions 

**1. Memory Issue with run_tracking.py** - Make sure to have sufficient memory  \
**2. Openmmlab Installations** - mmcv, mmpose, mmdet, mmyolo, mmengine - Make sure to use ```uncc-kubeflow-base``` image to avoid issues with mmcv. \
**3. mmcv,mmpose,mmdet,mmyolo version issues**
 - Make sure to follow above mentioned instructions for mmcv installation.
 - Clone the official repositories and replace the ones in the current versions(./mmpose, ./mmyolo)
 - Make sure to add back relevant missing files from the current repository into the latest cloned repositories.
