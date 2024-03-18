# CRAAC: Consistency Regularised Active Learning with Automatic Corrections for Real-life Road Image Annotations

Thank you for your interest in visiting our repo for the source code. 

# Installation:
1. Install detectron2 by Meta AI with instructions here:
https://github.com/facebookresearch/detectron2
2. Fork or clone our repo to your local computer.
3. Copy all the folders and main.py to the directory of your detectron2 repo
i.e. </path/to>/detectron2
4. We performed our experiments with conda environments in WSL2 (Ubuntu 22.04). Check if you have installed all the packages in your environment in environment.yaml

# Introductions to the CRAAC solution
Our experiments can be operated by the function calls in main.py.

Codes for each of the components can be found in ./src

Consistency Regularisation (CNS): adapted from the Consistency Regularisation algorithm for two-stage detectors by Jeong at al. (2019)
https://github.com/vlfom/CSD-detectron2

Scoring Module for Active Learning (AL): inspired by Yoo et al. (2019) and Wang et al. (2020).

Automatic Correction module (AC): main files in ./src/autocorr

# Key References
```
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}

@inproceedings{Jeong2019Consistency-basedDetection,
    title = {{Consistency-based Semi-supervised Learning for Object Detection}},
    year = {2019},
    booktitle = {Conference on Neural Information Processing Systems},
    author = {Jeong, Jisoo and Lee, Seungeui and Kim, Jeesoo and Kwak, Nojun},
    url = {https://github.com/soo89/CSD-SSD},
    address = {Vancouver, Canada},
    isbn = {9781713807933}
}

@inproceedings{Yoo2019LearningLearning,
    title = {{Learning Loss for Active Learning}},
    year = {2019},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    author = {Yoo, Donggeun and Kweon, In So},
    publisher = {IEEE},
    url = {http://arxiv.org/abs/1905.03677},
    address = {Long Beach, CA, USA},
    doi = {10.48550/arXiv.1905.03677},
    arxivId = {1905.03677}
}

@inproceedings{Wang2020Semi-supervisedPredictions,
    title = {{Semi-supervised Active Learning for Instance Segmentation via Scoring Predictions}},
    year = {2020},
    booktitle = {British Machine Vision Virtual Conference},
    author = {Wang, Jun and Wen, Shaoguo and Chen, Kaixing and Yu, Jianghua and Zhou, Xin and Gao, Peng and Li, Changsheng and Xie, Guotong},
    month = {12},
    url = {http://arxiv.org/abs/2012.04829},
    address = {Virtual},
    doi = {10.48550/arXiv.2012.04829},
    arxivId = {2012.04829}
}
```
