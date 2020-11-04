# SCRN
Code repository for our paper "Stacked Cross Refinement Network for Edge-Aware Salient Object Detection", ICCV 2019 poster. [Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Stacked_Cross_Refinement_Network_for_Edge-Aware_Salient_Object_Detection_ICCV_2019_paper.pdf) and [supplementary material](http://openaccess.thecvf.com/content_ICCV_2019/supplemental/Wu_Stacked_Cross_Refinement_ICCV_2019_supplemental.pdf) are available.

# Change Log
2020.11.4：We update the predicted saliency maps of 20 algorithms on SOC test and validation sets!
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
    * scipy
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
        * Download the pre-trained model from [google drive](https://drive.google.com/open?id=1PkGX9R-uTYpWBKX0lZRkE2qvvpz1-IiG) or [baidu yun](https://pan.baidu.com/s/1Gm-YptzsVnHU0a6YkdjQaQ) (code: ilhx), and put it in './model/'. This model is only trained on the training set of DUTS and tested on other datasets, including SOC and test set of DUTS. Set your dataset path, then
        ```
        python test_SCRN.py
        ```
        * You can also download the pre-computed saliency maps from [google drive](https://drive.google.com/open?id=1gRis5weSxuv9w6EZ23MPAnyDe-hUx07L) or [baidu yun](https://pan.baidu.com/s/1VHl_pWvbZGeAKgMwqFEHsw) (code: 8mty).

# SOC saliency maps
In the paper, we compare SCRN with nine methods on SOC validation set. Here we provide saliency maps of 20 SOD methods on both test and validation sets ([google drive](https://drive.google.com/file/d/10Jw1E4S6zQfeoa1SM3Aj93K0RnmAqRCg/view?usp=sharing) or [baidu yun](https://pan.baidu.com/s/1mWpE3jEVvGlb5VuSkSZ7jw) (code: wnjp)): [DSS](https://openaccess.thecvf.com/content_cvpr_2017/papers/Hou_Deeply_Supervised_Salient_CVPR_2017_paper.pdf)、[NLDF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Luo_Non-Local_Deep_Features_CVPR_2017_paper.pdf)、[SRM](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wang_A_Stagewise_Refinement_ICCV_2017_paper.pdf)、[Amulet](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_Amulet_Aggregating_Multi-Level_ICCV_2017_paper.pdf)、[DGRL](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Detect_Globally_Refine_CVPR_2018_paper.pdf)、[BMPM](https://openaccess.thecvf.com/content_cvpr_2018/papers_backup/Zhang_A_Bi-Directional_Message_CVPR_2018_paper.pdf)、[PiCANet-R](https://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_PiCANet_Learning_Pixel-Wise_CVPR_2018_paper.pdf)、[C2S-Net](https://openaccess.thecvf.com/content_ECCV_2018/papers/Xin_Li_Contour_Knowledge_Transfer_ECCV_2018_paper.pdf)、[RANet](https://openaccess.thecvf.com/content_ECCV_2018/papers/Shuhan_Chen_Reverse_Attention_for_ECCV_2018_paper.pdf)、[CPD](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_Cascaded_Partial_Decoder_for_Fast_and_Accurate_Salient_Object_Detection_CVPR_2019_paper.pdf)、[AFN](https://openaccess.thecvf.com/content_CVPR_2019/papers/Feng_Attentive_Feedback_Network_for_Boundary-Aware_Salient_Object_Detection_CVPR_2019_paper.pdf)、[BASNet](https://openaccess.thecvf.com/content_CVPR_2019/papers/Qin_BASNet_Boundary-Aware_Salient_Object_Detection_CVPR_2019_paper.pdf)、[PoolNet](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_A_Simple_Pooling-Based_Design_for_Real-Time_Salient_Object_Detection_CVPR_2019_paper.pdf)、[SCRN](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Stacked_Cross_Refinement_Network_for_Edge-Aware_Salient_Object_Detection_ICCV_2019_paper.pdf)、[SIBA](https://openaccess.thecvf.com/content_ICCV_2019/papers/Su_Selectivity_or_Invariance_Boundary-Aware_Salient_Object_Detection_ICCV_2019_paper.pdf)、[EGNet](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhao_EGNet_Edge_Guidance_Network_for_Salient_Object_Detection_ICCV_2019_paper.pdf)、[F3Net](https://aaai.org/ojs/index.php/AAAI/article/view/6916)、[GCPANet](https://aaai.org/ojs/index.php/AAAI/article/view/6633)、[MINet](https://openaccess.thecvf.com/content_CVPR_2020/papers/Pang_Multi-Scale_Interactive_Network_for_Salient_Object_Detection_CVPR_2020_paper.pdf).

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
If you have any question, please contact us (wuzh02@pcl.ac.cn).
