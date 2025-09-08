# MIT Sea Grant 


## ⚙️ Requirement
  * Platforms: Windows, Linux or maxOS
  * NVIDIA GPU (recommended for CV model training and inference).
  * python>=3.9 (tested on 3.12)
  * Pytorch with CUDA if has a CUDA-enabled GPU
  * [Ultralytic YOLO v11](https://github.com/ultralytics/ultralytics)
  * [Supervision](https://supervision.roboflow.com/latest/)
  * huggingface_hub (for downloading test dataset)

Setup the environment to run scripts in this repo (work inprogress..)
```
# clone the repo and enter directory
git clone https://github.com/zhongqic/river-herring-cv.git && cd river-herring-cv.git

# create a conda environment, with Python>=3.9
conda create -n river-herring-cv python=3.12

# Install requirements
pip install -r requirements.txt
```


### 📁 Annotated Dataset
Full set of bounding-box annotations from this project is available at [lila.science](https://lila.science/datasets/mit-sea-grant-river-herring/).      
It is also included in the [Community Fish Detection Dataset](https://lila.science/datasets/community-fish-detection-dataset/).  


Model training







