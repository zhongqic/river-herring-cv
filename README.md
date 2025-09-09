# MIT-Sea-Grant-River-Herring-Public
This repository contains code, models, tools, and documentation for a river herring monitoring system that uses underwater video and computer vision. It supports a complete workflow including: video processing and annotation, training custom YOLO-based detection models, tracking fish movements, generating fish counts and applying unbiased count corrections via importance sampling.


## ⚙️ Requirement
  * Platforms: Windows, Linux or maxOS
  * NVIDIA GPU (recommended for CV model training and inference).
  * python>=3.9 (tested on 3.12)
  * Pytorch with CUDA (v12.8) if has a CUDA-enabled GPU
  * [Ultralytic YOLO v11](https://github.com/ultralytics/ultralytics)
  * [Supervision](https://supervision.roboflow.com/latest/)
  * huggingface_hub (for downloading test dataset)

Setup the environment to run scripts in this repo (work inprogress..)
```
# clone the repo and enter directory
git clone https://github.com/zhongqic/river-herring-cv.git && cd river-herring-cv

# create a conda environment, with Python>=3.9
conda create -n river-herring-cv python=3.12

# Install requirements
pip install -r requirements.txt
```


### 📁 Annotated Dataset
Full set of bounding-box annotations from this project is available at [lila.science](https://lila.science/datasets/mit-sea-grant-river-herring/).      
It is also included in the [Community Fish Detection Dataset](https://lila.science/datasets/community-fish-detection-dataset/).  


## 🤖 YOLO model training

#### 1. Download test dataset (a small subset for testing)
```python
from huggingface_hub import hf_hub_download, snapshot_download
snapshot_download(repo_id="zhongqic/Fisheye-example", allow_patterns=["*.tar.gz", "data.yaml"], repo_type="dataset", local_dir="data/test_train")

```

```bash
# Unzip downloaded files
tar  -xzf data/test_train/train.tar.gz -C .\data\test_train
tar  -xzf data/test_train/val.tar.gz -C .\data\test_train
tar  -xzf data/test_train/test.tar.gz -C .\data\test_train
```

#### 2. YOLO model training
Ultralytics YOLO training is nicely packed and very easy to run.  Python scripts here can also be found in `src/train_yolo11.py`.  
```python

from ultralytics import YOLO

weights_path = "weights/yolo11l.pt"
yaml_file    = "data/test_train/data.yaml"

# Load a model
model = YOLO(weights_path) 
# Train the model
results = model.train(data=yaml_file, epochs=2, batch = 8)
# Validate on the test set
model.val(data  = yaml_file, split = "test")
```





