# Perspective+ Unet: Enhancing Segmentation with Bi-Path Fusion and Efficient Non-Local Attention for Superior Receptive Fields
[paper](https://ieeexplore.ieee.org/document/10447712) | [code](https://github.com/tljxyys/RDSTN_ultrasound)
***
>Abstract: _Precise segmentation of medical images is fundamental for extracting critical clinical information, which plays a pivotal role
in enhancing the accuracy of diagnoses, formulating effective treatment plans, and improving patient outcomes. Although Convolutional Neural
Networks (CNNs) and non-local attention methods have achieved notable success in medical image segmentation, they either struggle to capture
long-range spatial dependencies due to their reliance on local features, or face significant computational and feature integration challenges
when attempting to address this issue with global attention mechanisms. To overcome existing limitations in medical image segmentation, we propose
a novel architecture, Perspective+ Unet. This framework is characterized by three major innovations: (i) It introduces a dual-pathway strategy
at the encoder stage that combines the outcomes of traditional and dilated convolutions. This not only maintains the local receptive field but
also significantly expands it, enabling better comprehension of the global structure of images while retaining detail sensitivity. (ii) The framework
incorporates an efficient non-local transformer block, named ENLTB, which utilizes kernel function approximation for effective long-range dependency
capture with linear computational and spatial complexity. (iii) A Spatial Cross-Scale Integrator strategy is employed to merge global dependencies
and local contextual cues across model stages, meticulously refining features from various levels to harmonize global and local information.
Experimental results on the ACDC and Synapse datasets demonstrate the effectiveness of our proposed Perspective+ Unet. The code is available in the
supplementary material._
>
![image](https://github.com/tljxyys/RDSTN_ultrasound/blob/main/fig/Figure%201.png)
***
![image](https://github.com/tljxyys/RDSTN_ultrasound/blob/main/fig/Figure%202.png)
***
## Dependencies and Installation
* Clone this repo:
```
https://github.com/tljxyys/RDSTN_ultrasound.git
cd RDSTN_ultrasound
```
* Create a conda virtual environment and activate:
```
conda create -n rdstn python=3.7 -y
conda activate rdstn
```
* install necessary packages:
```
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
pip install timm==0.4.12
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8 pyyaml scipy
```
* Other requirements:
```
TensorboardX, yaml, numpy, tqdm, imageio
```
***
## Results
![image](https://github.com/tljxyys/RDSTN_ultrasound/blob/main/fig/Figure%203.png)
<img src="https://github.com/tljxyys/RDSTN_ultrasound/blob/main/fig/1215.gif" onload="this.onload=null;this.play();" /> <img src="https://github.com/tljxyys/RDSTN_ultrasound/blob/main/fig/1220.gif" onload="this.onload=null;this.play();" /> <img src="https://github.com/tljxyys/RDSTN_ultrasound/blob/main/fig/1222.gif" onload="this.onload=null;this.play();" />

The figures above are all gif file and will only play once. if you want to see the gif effect, please refresh the page.

## Bibtex
```
@INPROCEEDINGS{10447712,
  author={Hu, Jintong and Che, Hui and Li, Zishuo and Yang, Wenming},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Residual Dense Swin Transformer for Continuous Depth-Independent Ultrasound Imaging}, 
  year={2024},
  volume={},
  number={},
  pages={2280-2284},
  keywords={Image quality;Visualization;Ultrasonic imaging;Superresolution;Imaging;Streaming media;Transformers;Ultrasound imaging;Arbitrary-scale image super-resolution;Depth-independent imaging;Non-local implicit representation},
  doi={10.1109/ICASSP48485.2024.10447712}}
```
