# SCRN
Code repository for our paper "Stacked Cross Refinement Network for Edge-Aware Salient Object Detection", ICCV 2019 poster.

# Framework
![image](https://github.com/wuzhe71/SCAN/blob/master/figure/framework.png)

# Experiments
![results1](https://github.com/wuzhe71/SCAN/blob/master/figure/results1.png)

# Usage
1. Requirements
    * pytorch 0.40+

2. Clone the repo
```
git clone https://github.com/wuzhe71/SCRN.git 
cd SCRN
```

3. Train/Test
    * Train
        * Set your dataset path in train_SCRN.py and run it. 
    * Test
        * Download the pre-trained model from [google drive](https://drive.google.com/open?id=1PkGX9R-uTYpWBKX0lZRkE2qvvpz1-IiG) and put it in './model/'. Then set your dataset path in test_SCRN.py and run it.
        * You can also download he pre-computed saliency maps fron [google drive](https://drive.google.com/open?id=16nIFpcts43bmZdr9YxPT5x71f1cee6Of)

# if you think this code is helpful, please cite
```
@InProceedings{Wu_2019_ICCV,
author = {Wu, Zhe and Su, Li and Huang, Qingming},
title = {Stacked Cross Refinement Network for Edge-Aware Salient Object Detection},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}
```

# Contact Us
If you have any question, please contact us (zhe.wu@vipl.ict.ac.cn).
