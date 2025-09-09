# MIT-Sea-Grant-River-Herring-Public
This repository contains code, models, tools, and documentation for a river herring monitoring system that uses underwater video and computer vision. It supports a complete workflow including: video processing and annotation, training custom YOLO-based detection models, tracking fish movements, generating fish counts and applying unbiased count corrections via importance sampling.


### ⚙️ Requirement
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


#### 📁 Annotated Dataset
Full set of bounding-box annotations from this project is available at [lila.science](https://lila.science/datasets/mit-sea-grant-river-herring/).      
It is also included in the [Community Fish Detection Dataset](https://lila.science/datasets/community-fish-detection-dataset/).  


#### 🤖 YOLO model training

##### 1. Download test dataset (a small subset for testing)
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

##### 2. YOLO model training
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) model training is very easy to setup and run.  Python scripts here can also be found in `src/train_yolo11.py`.  
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

#### 🐟 Detect, Track and Count Fish
A yolo11 model pretrained on the full dataset is available under `weights/` for river herring detection and counting. The speed of processing each video mostly depending on GPU.


```python
python src/count_fish.py \
    data/raw_video/1_2024-05-07_10_06_48-355.mp4 \ # input video file
    weights/river-herring-yolo11.pt \              # model weight
    outputs/fish_count \                           # output dir for count results
    --class_id 0 \          # Class ID to count
    --save_video \          # include to save annotated video
    --tracker 'botsort.yaml' \  # tracker "bytetrack.yaml"
    --conf_thresh 0.8 \     # detection threshold
    --line_pos 0.6 \        # count line position, left - 0, right - 1
    --move_right "Upstream" \  # migration direction of fish swiming right
    --move_left "Downstream"   # migration direction of fish swiming left
```

**Batch processing**
To process multiple videos at once, list their file paths in `scripts/video_file_list.txt`. Then, run this bash script. The results for each video will be saved in a dedicated directory (named after the video file) within the specified output directory (outdir).

```bash
./scripts/batch_fish_counter.sh  scripts/video_file_list.txt weights/river-herring-yolo11.pt outputs/fish_count
```


#### 📈 DISCount for unbiased count estimate
In crowded scenes when fish overlap, low visibility, the cv model may miss detections or count incorrectly. We used the [DISCount approach](https://ojs.aaai.org/index.php/AAAI/article/view/30235), which is a detector-based importance sampling framework that integrates an imperfect detector with human-in-the-loop screening to produce unbiased estimates of object counts in CV tasks.

Detailed step-by-step tutorial on doing DISCount can be found [here](https://github.com/gperezs/DISCount/), and this [notebook](https://colab.research.google.com/drive/1bOEV7HCKZhJYfSGqCy47X0qPtgwCI85c?usp=sharing) with example from this River Herring project.



#### Links
<details><summary> <b>Expand</b> </summary>

|||
|---|---|
Fisheye Project website |https://www.woodwellclimate.org/project/fisheye/
Coonamessett River Trust |https://www.crivertrust.org   
Ipswich River Watershed Association |https://www.ipswichriver.org/   
MIT Sea Grant |https://seagrant.mit.edu/river-herring/
Huggingface test dataset |https://huggingface.co/zhongqic 


</details>

