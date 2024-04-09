# Perspective+ Unet: Enhancing Segmentation with Bi-Path Fusion and Efficient Non-Local Attention for Superior Receptive Fields
[paper]() | [code](https://github.com/tljxyys/Perspective-Unet) | [pretrained model]()
***
>Abstract: _Precise segmentation of medical images is fundamental for extracting critical clinical information, which plays a pivotal role
in enhancing the accuracy of diagnoses, formulating effective treatment plans, and improving patient outcomes. Although Convolutional Neural
Networks (CNNs) and non-local attention methods have achieved notable success in medical image segmentation, they either struggle to capture
long-range spatial dependencies due to their reliance on local features, or face significant computational and feature integration challenges
when attempting to address this issue with global attention mechanisms. To overcome existing limitations in medical image segmentation, we propose
a novel architecture, Perspective+ Unet. This framework is characterized by three major innovations: (i) It introduces a dual-pathway strategy
at the encoder stage that combines the outcomes of traditional and dilated convolutions. This not only maintains the local receptive field but
also significantly expands it, enabling better comprehension of the global structure of images while retaining detail sensitivity. (ii) The framework
incorporates an efficient non-local transformer block, named ENLTB, which utilizes kernel function approximation for effective long-range dependency
capture with linear computational and spatial complexity. (iii) A Spatial Cross-Scale Integrator strategy is employed to merge global dependencies
and local contextual cues across model stages, meticulously refining features from various levels to harmonize global and local information.
Experimental results on the ACDC and Synapse datasets demonstrate the effectiveness of our proposed Perspective+ Unet. The code is available in the
supplementary material._
>
![image](https://github.com/tljxyys/Perspective-Unet/blob/main/fig/model_architecture.png)
***
## 1. Dependencies and Installation
* Clone this repo:
```
https://github.com/tljxyys/Perspective-Unet.git
cd Perspective-Unet
```
* Create a conda virtual environment and activate:
```
conda create -n perspective_unet python=3.7 -y
conda activate perspective_unet
```
* install packages:
```
pip install -r requirements.txt
```
***
## 2. Data Preparation
The datasets we used are provided by TransUnet's authors. [Get processed data in this link] (https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd). If you would like to use the preprocessed data, please use it for research purposes and do not redistribute it (following the TransUnet's License). The directory structure of the whole project is as follows:
```
.
├── datasets
│   └──
├── lists
│   └── 
├── data
│   ├── Synapse
│   │   ├── train_npz
│   │   │   ├── case0005_slice000.npz
│   │   │   └── *.npz
│   │   └── test_vol_h5
│   │       ├── case0001.npy.h5
│   │       └── *.npy.h5
│   └── ACDC
│       ├── train
│       │   ├── case_001_sliceED_0.npz
│       │   └── *.npz
│       ├── test
│       │   ├── case_002_volume_ED.npz
│       │   └── *.npz
│       └── train
│           ├── case_019_sliceED_0.npz
│           └── *.npz
├── networks
│   └── 
├── train
├── test
└── trainer
```
## Results
| First Header  | Second Header |
| ------------- | ------------- |
| Content Cell  | Content Cell  |
| Content Cell  | Content Cell  |
***
![image](https://github.com/tljxyys/Perspective-Unet/blob/main/fig/visual_results.png)

