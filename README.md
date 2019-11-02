# SCRN
Code repository for our paper "Stacked Cross Refinement Network for Edge-Aware Salient Object Detection", ICCV 2019 poster. [Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Stacked_Cross_Refinement_Network_for_Edge-Aware_Salient_Object_Detection_ICCV_2019_paper.pdf) and [supplementary material](http://openaccess.thecvf.com/content_ICCV_2019/supplemental/Wu_Stacked_Cross_Refinement_ICCV_2019_supplemental.pdf) are available.

# Framework
![image](https://github.com/wuzhe71/SCAN/blob/master/figure/framework.png)

# Experiments
1. Results on traditional datasets
![results1](https://github.com/wuzhe71/SCAN/blob/master/figure/results1.png)

2. Results on SOC (attribute-based performance, structure simalarity scores), more comparison can be found in [SOC Leaderboard](http://dpfan.net/SOCBenchmark/)
![results3](https://github.com/wuzhe71/SCAN/blob/master/figure/results3.png)

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
        * Download datasets: [DUTS](http://saliencydetection.net/duts/), [DUT-OMRON](http://saliencydetection.net/dut-omron/), [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), [HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html), [PASCAL-S](http://www.cbi.gatech.edu/salobj/), [THUR15K](https://mmcheng.net/gsal/), [SOC](http://dpfan.net/SOCBenchmark/)
        * Set your dataset path, then
        ```
        python train_SCRN.py
        ```
        * We only use multi-scale traing for data agumentation, and the lr is set as 0.002. If you change to single-scale training, the lr should better change to 0.005.
    * Test
        * Download the pre-trained model from [google drive](https://drive.google.com/open?id=1PkGX9R-uTYpWBKX0lZRkE2qvvpz1-IiG) or [baidu yun](https://pan.baidu.com/s/1-sAObg4cegWLF7ZZvhYO0A) (code: ozjr), and put it in './model/'. This model is only trained on the training set of DUTS and tested on other datasets, including SOC and test set of DUTS. Set your dataset path, then
        ```
        python test_SCRN.py
        ```
        * You can also download he pre-computed saliency maps fron [google drive](https://drive.google.com/open?id=1gRis5weSxuv9w6EZ23MPAnyDe-hUx07L) or [baidu yun](https://pan.baidu.com/s/1YAvKOjFNE22DnbXOIoJ7LQ) (code: cld7)

# Citation
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
