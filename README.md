# Edge-Aware Fine-Grained Classification using Transfer Learning

## Table of contents

1. [Overview](#overview)
2. [Credits](#credits)
3. [Dependencies and Installation](#dependencies-and-installation)
4. [Dataset Preparation](#dataset-preparation)
5. [Dataset visualization](#dataset-visualization)
6. [Training & Evaluation](#training--evaluation)
7. [Inference demo](#inference-demo)

## Overview
This project explores the application of transfer learning on a small subset of the iNaturalist 2017 dataset, aiming to optimize fine-grained species classification for edge devices.

### Objectives
- Fine-tuned the model on a regional species subset
- Implement an 'Other' classification for unrecognized species
- Optimize the `logits` layer (final characterization) layer to improve inference performance
- Analyze the feature maps of the model if fine-tuning proves insufficient
- Improve real-time performance & efficiency of classification models

### Why Edge computing?
In real-world scenario, deploying large-scale models on edge devices (RaspberryPi, IoT devices, ...) is challenging due to:
- Limited computational power
- Lower memory availability
- Network independence

By fine-tuning the model with a focused dataset and optimizing its feature representations, we aim to enhance model performance while minimizing resource consumption.

## Credits 
Authors of the original works this project based on:

- [Yin Cui](http://www.cs.cornell.edu/~ycui/)
- [Yang Song](https://ai.google/research/people/author38269)
- [Chen Sun](http://chensun.me/)
- Andrew Howard
- [Serge Belongie](http://blogs.cornell.edu/techfaculty/serge-belongie/)

CVPR2018

**This project is based on the original work from:**
- [Large Scale Fine-Grained Categorization and Domain-Specific Transfer Learning (CVPR 2018)](https://arxiv.org/abs/1806.06193)
- [Original GitHub Repository](https://github.com/richardaecn/cvpr17-inaturalist-transfer)

## Dependencies and Installation
Setting up TensorFlow 1.x with GPU acceleration is challenging due to outdated CUDA and cuDNN requirements. The original repository used Python 3.5.6, but due to dependency issues (pip, OpenSSL, etc.), Python 3.6.13 was found to be the most stable version.

### Recommended Setup
- Python 3.6.13 via Anaconda
- `TensorFlow` 1.11
- Nvidia libraries installed via Conda
    - `cuda-nvcc` 11.3.58
    - `cudatoolkit` 9.0
    - `cudnn` 7.6.5
- Virtual environment using Conda instead of Pyenv

### 1. Install Anaconda (If not installed)

Follow their guide here: [https://www.anaconda.com/docs/getting-started/anaconda/install](https://www.anaconda.com/docs/getting-started/anaconda/install)

### 2. Create a Conda virtual environment

```bash
conda create --name cvpr18 python=3.6.13 -y
conda activate cvpr18
```

### 3. Install Nvidia packages

```bash
conda install cudatoolkit=9.0 cudnn==7.6.5 cuda-nvcc==11.3.58 -y
```

### 4. Create a virtual python environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 5. Install TensorFlow 1.11

You can either install through `pip` or download the prebuilt wheel on [pypi.org](https://pypi.org/project/tensorflow-gpu/1.11.0/#files). Make sure you download the `cp36-arch.whl` where `arch` is your OS of choice.

If no GPU is available, you can install a tensorflow build that is optimized for new CPU instruction here: [lakshayg/tensorflow-build](https://github.com/lakshayg/tensorflow-build)

```bash
pip install tensorflow-gpu==1.11
```
Or
```bash
pip install tensorflow_gpu-1.11.0-cp36-cp36m-manylinux1_x86_64.whl 
```

### 6. Install required dependencies

```bash
pip install numpy scipy matplotlib pillow scikit-learn scikit-image pyemd
```

### 7. Verify if TensorFlow has been installed successfully

```bash
python -c "import tensorflow as tf; print(tf.test.is_gpu_available())"
```

## Dataset Preparation
### TODO: Automating Dataset Preparation
Currently, dataset preparation requires manually setting paths in multiple scripts, which can be error-prone.  
Planned Improvement: Introduce an orchestrator script to automate:
- Running all preprocessing steps sequentially.
- Handling dataset paths dynamically.
- Reducing manual intervention.

This project uses a subset of iNaturelist 2017 combined with species from Haute-Garonne. The dataset must be manually downloaded and processed before training.

### 1. Download and extract iNaturelist 2017
[https://github.com/visipedia/inat_comp/tree/master/2017](https://github.com/visipedia/inat_comp/tree/master/2017)

We only uses the data in `Aves/` and `Insecta/`.
Extract the dataset with the program of your choice to `data/inat2017`

### 2. Run the web crawler to obtain the dataset from Haute-Garonne

```bash
python scripts/haute_garonne_web_crawl.py
```
This generates `output/iNaturalist_All_Species_Full.json`

### 3. Run `scripts/dataset_analyzer.py` on both datasets to extract their properties
```bash
cd scripts
python dataset_analyzer.py ../data/inat2017 # path to inat2017
python dataset_analyzer.py output/iNaturalist_All_Species_Full.json # path to the HG JSON file
```

### 4. Match Species Between Datasets

This step uses `cross_reference.py` to find common species between iNaturelist 2017 and Haute-Garonne

Modify `FILE_1` and `FILE_2` to the correspond directory of the JSON file:
- `FILE_1`: iNaturelist 2017 Aves and Insecta
- `FILE_2`: Haute-Garonne Aves and Insecta

```bash
cd scripts
python scripts/cross_reference.py
```

This generates `output/matched_species_Aves_Insecta.json`

### 5. Copy matched species
This creates the Haute-Garonne dataset that we can use for training later.

Modify `SRC_DATASET` and `DST_DATASET` to the correspond directory of the dataset:
- `SRC_DATASET`: the base directory of `inat2017`
- `DST_DATASET`: the destination directory of `haute-garonne`

```bash
python scripts/copy_matched_species.py
```
This will create a new dataset under location: `DST_DATASET`

### 6. Handle "Other" classification

This script needs to be run in the base directory of the repo.
The model needs an "Other" class to detect unknown species:
- `INAT_DIR`: The path of `inat2017` dataset
- `HG_DIR`: The path of `haute-garonne` dataset
- `OUTPUT_DIR`: The path of the dataset with "Other" classification
- `OTHER_DIR`: The path of "Other", default to `OTHER_DIR/Other`

```bash
python scripts/other_classifier.py
```

### 7. Generate Dataset Manifest

Before converting to TFRecords, we create `train.txt` and `val.txt`

```bash
python scripts/train_val_splitter.py
```
This script will produce:
- `train.txt`: list of training images
- `val.txt`: list of validation images
- `dataset_manifest.txt`: a list of all image paths and their classification
- `dataset_species_labels.json`: a JSON file containing the species and their classification

This script `stdout` output will be used for the metadata of `slim/datasets/dataset_factory_fgvc.py` later. **Take note of it!**

### 8. Convert the dataset to TFRecords

```bash
python python convert_dataset.py --dataset_name=inat2017_other --num_shards=10
```

## Dataset visualization

You can run `scripts/visualizer.py` to visualize the dataset. This script produces
- A venn-diagram to visualize the intersection of two datasets.
- A class composition bar chart to visualize the number of images per species per class.
- A CDF analysis of a class of species in the dataset.

## Training & Evaluation

### Add the new dataset to `dataset_factory_fgvc.py`

Add the following entry to `datasets_map`:
```py
'inat2017_other': {'num_samples': {'train': 318668, 'validation': 35408},
                   'num_classes': 286},
```
The values for `num_samples` and `num_classes` are obtained from `train_val_splitter.py` above.

### Start the training

Modify the variables in `train.sh` to match the new dataset before continuing.
The new model checkpoints will be saved under `checkpoints/{dataset_name}/`

```bash
./train.sh
```

### Monitor the training

```bash
tensorboard --logdir=./checkpoints/inat2017_other/ --port=6006
```

### Evaluate the model

Modify the variables in `eval.sh` to match the new dataset before continuing.
The evaluation script will load the checkpoints at `checkpoints/{dataset_name}/`
Outputs accuracy and Recall@5

```bash
./eval.sh
```

## Inference demo

Here's are the predefined global variables that you need to modify in order to use this script:

```py
MODEL_DIR = "The model checkpoints path"
IMAGE_PATH = "Absolute path to the image you want to perform inference on"
CLASS_MANIFEST_PATH = "dataset_species_labels.json path that we created with train_val_splitter.json"
IMAGE_SIZE = 299  
NUM_CLASSES =  286 # Number of species of the dataset (entire Other class is count as 1)
LABELS_OFFSET = 0
```
Then run this command:

```bash
python scripts/inference/inference.py
```