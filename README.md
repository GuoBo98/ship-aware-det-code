# Ship-aware-det

This is the code of Ship-aware-det. The project is based on the OBBDetection.

We will further consolidating and optimizing our code as soon as possible.

The MCSD Dataset is under uploading, I will update the download link here once it is uploaded.

## introduction

OBBDetection is an oriented object detection toolbox modified from [MMdetection v2.2](https://github.com/open-mmlab/mmdetection).

## Installation

Please refer to [install.md](docs/install.md) for installation and dataset preparation.

## Get Started

### Oriented models training and testing

If you want to train or test a oriented model, please refer to [oriented_model_starting.md](docs/oriented_model_starting.md).

### How to use MMDetection

If you are not familiar with MMdetection, please see [getting_started.md](docs/getting_started.md) for the basic usage of MMDetection. There are also tutorials for [finetuning models](docs/tutorials/finetune.md), [adding new dataset](docs/tutorials/new_dataset.md), [designing data pipeline](docs/tutorials/data_pipeline.md), and [adding new modules](docs/tutorials/new_modules.md).

## Acknowledgement

We refered [S2ANet](https://github.com/csuhan/s2anet) and [AerialDetection](https://github.com/dingjiansw101/AerialDetection) when develping OBBDetection.

This toolbox is modified from [MMdetection](https://github.com/open-mmlab/mmdetection). If you use this toolbox or benchmark in your research, please cite the following information.

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal = {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```

This is the official implement of [Oriented R-CNN](configs/obb/oriented_rcnn). if it is used in your research, please cite the following information.

```
@InProceedings{Xie_2021_ICCV,
  author = {Xie, Xingxing and Cheng, Gong and Wang, Jiabao and Yao, Xiwen and Han, Junwei},
  title = {Oriented R-CNN for Object Detection},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month = {October},
  year = {2021},
  pages = {3520-3529} }
```
