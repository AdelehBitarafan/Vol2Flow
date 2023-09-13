# Vol2Flow: Segment 3D Volumes using a Sequence of Registration Flows

This repository contains the codes (in PyTorch) for [Vol2Flow: Segment 3D Volumes using a Sequence of Registration Flows](https://link.springer.com/chapter/10.1007/978-3-031-16440-8_58) published in [MICCAI, 2022] (https://conferences.miccai.org/2022/en/).

![FinalLearning](https://user-images.githubusercontent.com/70052073/176658628-fcca260a-73c0-4388-9434-2bbf72fe11d8.png)


**Abstract:**

This work proposes a self-supervised algorithm to segment each arbitrary anatomical structure in a 3D medical image produced under various acquisition conditions, dealing with domain shift problems and generalizability. Furthermore, we advocate an interactive setting in the inference time, where the self-supervised model trained on unlabeled volumes should be directly applicable to segment each test volume given the user-provided single slice annotation. To this end, we learn a novel 3D registration network, namely Vol2Flow, from the perspective of image sequence registration to find 2D displacement fields between all adjacent slices within a 3D medical volume together. Specifically, we present a novel 3D CNN-based architecture that finds a series of registration flows between consecutive slices within a whole volume, resulting in a dense displacement field. A new self-supervised algorithm is proposed to learn the transformations or registration fields between the series of 2D images of a 3D volume. Consequently, we enable gradually propagating the userprovided single slice annotation to other slices of a volume in the inference time. We demonstrate that our model substantially outperforms related methods on various medical image segmentation tasks through several experiments on different medical image segmentation datasets.

# Usage
 The main file is "train.py". It contains the learning Vol2Flow in an unsupervised manner. 

```
python train.py
```

# Bug Report

If you find a bug, please send a bug report to adeleh.bitarafan[at]sharif.edu. You can also send me any comment or suggestion about the program.


# Cite
If you find this code useful, please cite our paper. Thanks!

```
@inproceedings{bitarafan2022vol2flow,
  title={Vol2Flow: Segment 3D Volumes Using a Sequence of Registration Flows},
  author={Bitarafan, Adeleh and Azampour, Mohammad Farid and Bakhtari, Kian and Soleymani Baghshah, Mahdieh and Keicher, Matthias and Navab, Nassir},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={609--618},
  year={2022},
  organization={Springer}
}
```


