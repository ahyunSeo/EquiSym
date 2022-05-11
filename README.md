## Reflection and Rotation Detection via Equivariant Learning (CVPR 2022)

<p align="center">
Ahyun Seo, Byungjin Kim, Suha Kwak, Minsu Cho
</p>

<p align="center">
    <a href="https://arxiv.org/abs/2203.16787">[paper]</a>
    <a href="http://cvlab.postech.ac.kr/research/EquiSym">[project page]</a>
</p>

Official PyTorch implementation of *Reflection and Rotation Detection via Equivariant Learning (CVPR 2022)*.

### Environment
```
    conda create --name EquiSym python=3.7
    conda activate EquiSym
    conda install pytorch==1.7.0 torchvision==0.8.1 cudatoolkit=11.0 -c pytorch
    conda install -c conda-forge matplotlib
    pip install albumentations=0.5.2 shapely opencv-python tqdm
    
    mkdir weights wandb sym_datasets

```

### Datasets and weights
- download DENDI [onedrive](https://postechackr-my.sharepoint.com/:u:/g/personal/lastborn94_postech_ac_kr/ES2ftVVmTc5Du78EBgfTGy8BwygV_HRa5nWciYeq3cTvoQ?e=y9ETja) or [DENDI](https://github.com/ahyunSeo/DENDI)
- trained [weights](https://postechackr-my.sharepoint.com/:u:/g/personal/lastborn94_postech_ac_kr/EbHHT8lIPThPhYcjU2dLbucBT6jfcNDilC7UXjlSDGKXtA?e=FxdyYk): EquiSym(ours), EquiSym(CNN ver.), pre-trained ReResNet50(D8)

```
.
├── sym_datasets
│   └── DENDI
│       ├── symmetry
│       ├── symmetry_polygon
│       ├── reflection_split.pt
│       ├── rotation_split.pt
│       └── joint_split.pt
├── weights
│   ├── v_equiv_aux_ref_best_checkpoint.pt
│   ├── v_equiv_aux_rot_best_checkpoint.pt
│   ├── v_cnn_ref_best_checkpoint.pt
│   ├── v_cnn_rot_best_checkpoint.pt
│   └── re_resnet50_custom_d8_batch_512.pth
├── (...) 
└── main.py
```

### Demo & Test
- visualize results using the input images in ./imgs

```
    python demo.py --ver equiv_aux_ref -rot 0 -eq --get_theta 10  
    python demo.py --ver equiv_aux_rot -rot 1 -eq --get_theta 10 
```

- test with pretrained weights

```
    python train.py --ver equiv_aux_ref -t -rot 0 -eq -wf --get_theta 10 
    python train.py --ver equiv_aux_rot -t -rot 1 -eq -wf --get_theta 10 
```

- vis(test) with pretrained weights of vanilla CNN model

```
    python demo.py --ver cnn_ref -rot 0
    python demo.py --ver cnn_rot -rot 1
```


### Training
The trained weights and arguments will be save to the checkpoint path corresponding to the VERSION_NAME.

```
    python train.py --ver VERSION_NAME_REF -tlw 0.01 --get_theta 10 -rot 0 -eq
    python train.py --ver VERSION_NAME_ROT -tlw 0.001 --get_theta 10 -rot 1 -eq
```


### References
- ReDet [link](https://github.com/csuhan/ReDet)
- e2cnn [link](https://github.com/QUVA-Lab/e2cnn)
- pytorch-deeplab-xception [link](https://github.com/jfzhang95/pytorch-deeplab-xception)
- PMCNet [link](https://github.com/ahyunSeo/PMCNet)


### Citation
If you find our code or paper useful to your research work, please consider citing:
```
@inproceedings{seo2022equisym,
    author   = {Seo, Ahyun and Kim, Byungjin and Kwak, Suha and Cho, Minsu},
    title    = {Reflection and Rotation Symmetry Detection via Equivariant Learning},
    booktitle= {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year     = {2022}
}
```
