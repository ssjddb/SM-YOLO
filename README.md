# 1. Code Dependencies and Requirements

## 1.1 Experimental Platform and Environment

- **CPU**: Intel Xeon Platinum 8369B 
- **GPU**: NVIDIA A10  
- **Python**: 3.9.0
- **Pytorch**: 2.1.0
- **CUDA**: 11.8 

## 1.2 Training Hyperparameters

- **Decay strategy**: cosine  
- **Optimizer**: SGD  
- **Learning rate**: 0.01
- **Weight decay**: 0.0005
- **Momentum**: 0.937
- **Close mosaic**: 10 
- **Total epochs**: 500  
- **Batch size**: 64
- **Patience**: 100 

## 1.3 Requirements

- **faster-coco-eval>=1.6.5**
- **PyYAML**
- **tensorboard**
- **scipy**
- **calflops**
- **thop**
- **transformers**
- **pytorch_wavelets==1.3.0**
- **timm==1.0.7**
- **grad-cam==1.5.4**
- **tidecv**
- **einops**

---

# 2. File Description

- **Model configuration file**: `ultralytics/yaml/v8-n-SKS-MFF`  
- **train.py**: Script for training the model  
- **track.py**: Inference script  
- **test.py**: Testing script
- **FPS.py**: Script for calculating model inference time and FPS  
- **GFLOPS.py**: Script for calculating model storage size  

---

# 3. Key Modules of the Algorithm

## 3.1 Spatial Kernel Selection (SKS)

**Code**: `ultralytics/nn/modules/SKS.py`

**Description**: Spatial Kernel Selection (SKS) module, which enables the network to dynamically adjust its receptive field according to defect scales, thereby enabling effective extraction of multi-scale feature representations. 

## 3.2 Multi-scale Feature Fusion (MFF)

**Code**: `ultralytics/nn/modules/MFF.py`

**Description**: Multi-scale Feature Fusion (MFF) module combines shallow and deep contextual information, minimizing information loss and enhancing feature representation.  

---

# 4. Related Details

This paper conducts experiments using three popular public datasets: NEU-DET, GC10-DET, and APDDD. 

All datasets are divided into training, validation, and test sets in an 8:1:1 ratio. 

The relevant datasets can be obtained from the Baidu Netdisk link:  
[Baidu Netdisk Link](https://pan.baidu.com/s/1vROCvEXNuEyt3P8PnneYPQ)  
Extraction code: `u5rs`  

If you have any questions or needs, please contact me:  
Email: [xs1113@stu.ahjzu.edu.cn](mailto:xs1113@stu.ahjzu.edu.cn)  

---

# 5. Citation Format

**Article title**: Steel Surface Defect Detection Based on Dynamic Receptive Field and Multi-Scale Features Fusion

**Submitted journal**: The Visual Computer