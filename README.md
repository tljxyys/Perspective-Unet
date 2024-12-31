# Perspective+ Unet: Enhancing Segmentation with Bi-Path Fusion and Efficient Non-Local Attention for Superior Receptive Fields🚀

[![](https://img.shields.io/badge/Project-Page-green.svg)](https://github.com/tljxyys/Perspective-Unet) [![](https://img.shields.io/badge/Paper-ArXiv-red.svg)](https://arxiv.org/abs/2406.14052) [![](https://img.shields.io/badge/Pretrained-🏁Replicate-blue.svg)](https://drive.google.com/drive/folders/1IYZjQWIdCBWFT8fZFC_NQ-sACQH-CNOE?usp=sharing) [![](https://img.shields.io/badge/Dataset-🔰Synapse-blue.svg)](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd) [![](https://img.shields.io/badge/Dataset-🔰ACDC-blue.svg)](https://drive.google.com/file/d/1s3efWoWd2xojorXyjzBXb9ECtIiwBt1P/view?usp=sharing)

>**Abstract**: _Precise segmentation of medical images is fundamental for extracting critical clinical information, which plays a pivotal role
in enhancing the accuracy of diagnoses, formulating effective treatment plans, and improving patient outcomes. Although Convolutional Neural
Networks (CNNs) and non-local attention methods have achieved notable success in medical image segmentation, they either struggle to capture
long-range spatial dependencies due to their reliance on local features, or face significant computational and feature integration challenges
when attempting to address this issue with global attention mechanisms. To overcome existing limitations in medical image segmentation, we propose
a novel architecture, Perspective+ Unet. This framework is characterized by three major innovations: **(i)** It introduces a dual-pathway strategy
at the encoder stage that combines the outcomes of traditional and dilated convolutions. This not only maintains the local receptive field but
also significantly expands it, enabling better comprehension of the global structure of images while retaining detail sensitivity. **(ii)** The framework
incorporates an efficient non-local transformer block, named ENLTB, which utilizes kernel function approximation for effective long-range dependency
capture with linear computational and spatial complexity. **(iii)** A Spatial Cross-Scale Integrator strategy is employed to merge global dependencies
and local contextual cues across model stages, meticulously refining features from various levels to harmonize global and local information.
Experimental results on the ACDC and Synapse datasets demonstrate the effectiveness of our proposed Perspective+ Unet._
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
* Install packages:
```
pip install -r requirements.txt
```

## 2. Data Preparation
- The Synapse dataset we used are provided by TransUnet's authors. [![](https://img.shields.io/badge/Dataset-🚀Synapse-blue.svg)](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd). If you would like to use the preprocessed data, please use it for research purposes and do not redistribute it (following the TransUnet's License). The ACDC dataset can be obtained from [![](https://img.shields.io/badge/Dataset-🚀ACDC-blue.svg)](https://www.creatis.insa-lyon.fr/Challenge/acdc/).
- I am not the owner of these two preprocessed datasets. **Please follow the instructions and regulations set by the official releaser of these two datasets.** The directory structure of the whole project is as follows:
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

## 3. Training
- Run the train script on synapse dataset. The batch size and epoch we used is 12 and 600, respectively.
```
python train.py --dataset Synapse --output_dir './model_output_Synapse' --max_epochs 600 --batch_size 12
```
- Run the train script on ACDC dataset. The batch size and epoch we used is 12 and 1000, respectively.
```
python train.py --dataset Synapse --output_dir './model_output_ACDC' --max_epochs 1000 --batch_size 12
```

## 4. Testing
- Download the pretrained model for inference. [Get pretrained model in this link] [![](https://img.shields.io/badge/Pretrained-🚀Replicate-blue.svg)](https://drive.google.com/drive/folders/1IYZjQWIdCBWFT8fZFC_NQ-sACQH-CNOE?usp=sharing). **Please save the .pth file in the ./model_output_Synapse or ./model_output_ACDC**.
```
python test.py --dataset Synapse --is_saveni True --output_dir './model_output_Synapse' --max_epoch 600 --batch_size 12 --test_save_dir './model_output_Synapse/predictions'
```
```
python test.py --dataset ACDC --is_saveni True --output_dir './model_output_ACDC' --max_epoch 1000 --batch_size 12 --test_save_dir './model_output_ACDC/predictions'
```

## 5. Results
- Segmentation accuracy of different methods on the Synapse multi-organ CT
dataset. The best results are shown in **bold**.

| Methods | DSC⬆️ | HD⬇️ | Aorta | Gallbladder | Kidney(L) | Kidney(R) | Liver | Pancreas | Spleen | Stomach |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U-Net | 76.85 | 39.70 | 89.07 | 69.72 | 77.77 | 68.60 | 93.43 | 53.98 | 86.67 | 75.58 |
| R50 Att-UNet | 75.57 | 36.97 | 55.92 | 63.91 | 79.20 | 72.71 | 93.56 | 49.37 | 87.19 | 74.95 |
| Att-UNet | 77.77 | 36.02 | 89.55 | 68.88 | 77.98 | 71.11 | 93.57 | 58.04 | 87.30 | 75.75 |
| R50 ViT | 71.29 | 32.87 | 73.73 | 55.13 | 75.80 | 72.20 | 91.51 | 45.99 | 81.99 | 73.95 |
| TransUnet | 77.48 | 31.69 | 87.23 | 63.13 | 81.87 | 77.02 | 94.08 | 55.86 | 85.08 | 75.62 |
| SwinUNet | 79.12 | 21.55 | 85.47 | 66.53 | 83.28 | 79.61 | 94.29 | 56.58 | 90.66 | 76.60 |
| AFTer-UNet | 81.02 | - | __90.91__ | 64.81 | 87.90 | 85.30 | 92.20 | 63.54 | 90.99 | 72.48 | 
| ScaleFormer | 82.86 | 16.81 | 88.73 | __74.97__ | 86.36 | 83.31 | 95.12 | 64.85 | 89.40 | 80.14 |
| MISSFormer | 81.96 | 18.20 | 86.99 | 68.65 | 85.21 | 82.00 | 94.41 | 65.67 | 91.92 | 80.81 |
| FCT | 83.53 | - | 89.85 | 72.73 | __88.45__ | __86.60__ | __95.62__ | 66.25 | 89.77 | 79.42 |
| MSAANet | 82.85 | 18.54 | 89.40 | 73.20 | 84.31 | 78.53 | 95.10 | 68.85 | 91.60 | 81.78 |
| __Perspective+ (Ours)__ | __84.63__ | __11.74__ | 89.38 | 70.80 | 87.57 | 85.78 | 95.30 | __70.71__ | __94.41__ | __83.06__ |

- Segmentation accuracy of different methods on the ACDC dataset. The best
results are shown in **bold**.

| Methods | DSC⬆️ | RV | Myo | LV |
| --- | --- | --- | --- | --- |
| R50 U-Net | 87.55 | 87.10 | 80.63 | 94.92 |
| R50 Att-UNet | 86.75 | 87.58 | 79.20 | 93.47 |
| R50 ViT | 87.57 | 86.07 | 81.88 | 94.75 |
| TransUNet | 89.71 | 88.86 | 84.53 | 95.73 |
| SwinUNet | 90.00 | 88.55 | 85.62 | 95.83 |
| ScaleFormer | 90.17 | 87.33 | 88.16 | 95.04 |
| UNETR | 88.61 | 85.29 | 86.52 | 94.02 |
| MCTE | 91.31 | 89.14 | 89.51 | 95.27 |
| MISSFormer | 91.19 | 89.85 | 88.38 | 95.34 |
| nnFormer | 92.06 | 90.94 | 89.58 | 95.65 |
| **Perspective+ (Ours)** | **92.54** | **90.92** | **90.49** | **96.20** |

- Ablation study on the impact of modules. BPRB: Bi-Path Residual Block, ENLTB: Efficient Non-Local Transformer Block, SCSI: Spatial Cross-Scale Integrator. 
  
| Model | BPRS | SCSI | ENLTB | DSC⬆️ | HD⬇️ |
| --- | --- | --- | --- | --- | --- |
| Setting 1 | ✖️ | ✖️ | ✖️ | 84.04 | 16.63 |
| Setting 2 | ✖️ | ✖️ | ✔️ | 84.55 | 13.60 |
| Setting 3 | ✖️ | ✔️ | ✖️ | 83.79 | 21.71 |
| Setting 4 | ✖️ | ✔️ | ✔️ | 84.35 | 17.88 |
| Setting 5 | ✔️ | ✖️ | ✖️ | 83.36 | 14.70 |
| Setting 6 | ✔️ | ✖️ | ✔️ | 85.07 | 14.60 |
| Setting 7 | ✔️ | ✔️ | ✖️ | 83.92 | 13.94 |
| Setting 8 | ✔️ | ✔️ | ✔️ | 84.63 | 11.74 | 


- Visualized segmentation results of different methods on the Synapse multi-organ CT dataset. Our method (the last column) exhibits the smoothest boundaries
and the most accurate segmentation outcomes.

![image](https://github.com/tljxyys/Perspective-Unet/blob/main/fig/visual_results.png)

- Visualization of attention heat maps from the intermediate layers of the network. Highlighting areas are closely aligned with segmentation labels, demonstrating
our Perspective+ Unet’s accuracy in feature identification and localization.

![image](https://github.com/tljxyys/Perspective-Unet/blob/main/fig/intermediate.png)

## 6. Reference
- [TransUnet](https://github.com/Beckschen/TransUNet)
- [SwinUnet](https://github.com/microsoft/Swin-Transformer)
- [MissFormer](https://github.com/ZhifangDeng/MISSFormer)

## 7. Bibtex
```
@inproceedings{hu2024perspective+,
  title={Perspective+ Unet: Enhancing Segmentation with Bi-Path Fusion and Efficient Non-Local Attention for Superior Receptive Fields},
  author={Hu, Jintong and Chen, Siyan and Pan, Zhiyi and Zeng, Sen and Yang, Wenming},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={499--509},
  year={2024},
  organization={Springer}
}
```
