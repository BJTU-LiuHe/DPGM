# “Deep probabilistic graph matching

## Requirements
tensorflow        1.14.0  
dm-sonnet         1.23

## Datasets
#### Pascal VOC Keypoint 
1. Download [VOC2011 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2011/index.html) and make sure it looks like `data/PascalVOC/VOC2011` 
2. Download [keypoint annotation](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/poselets/voc2011_keypoints_Feb2012.tgz) for VOC2011 and make sure it looks like `data/PascalVOC/annotations`
3. The train/test split is available in `data/PascalVOC/voc2011_pairs.npz`
  
#### Willow Objects 
1. Download [Willow-ObjectClass](http://www.di.ens.fr/willow/research/graphlearning/WILLOW-ObjectClass_dataset.zip) dataset
2. Unzip the dataset and make sure it looks like `data/Willow`

#### SPair-71k
1. Download [SPair-71k](http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz) dataset
2. Unzip the dataset and make sure it looks like `data/SPair71k`

In implementation, the keypoint features are extracted by the standard VGG16-bn backbone networks in advance. The raw keypoint features for the three datasets are packed in the [KPT_features.zip](https://drive.google.com/file/d/14iApmo8u0XJ81-3OIz6Y-tVZAJaQokTT/view?usp=sharing). Unzip the package file and make sure it looks like `data/KPT_features`

## Experiments
### Training
To train the model on different datasets, please change the configurations in GM_GenData.py and run  
`python GM_Train.py`
