# Unofficial implement of Live3DPortrait
This is an unofficial implement of [Live3DPortrait: Real-Time Radiance Fields for Single-Image Portrait View Synthesis](https://research.nvidia.com/labs/nxp/lp3d/) in Pytorch. 

![display1](./docs/display1.png)
![display2](./docs/display2.png)
![display3](./docs/display3.png)

https://github.com/Dong142857/Live3DPortrait/assets/47628302/d00b104c-3a9c-4ce0-bad7-379ed85a7e8e




## Pretrained Model
The checkpoints can be downloaded from [here](https://drive.google.com/drive/folders/1Z6uri8pH048Qzyhu0PE3llxO5EC7ic_v?usp=drive_link). 
<!-- InceptionV3 pretrained model comes from [here](https://github.com/mseitzer/pytorch-fid/releases). -->
Place them under `./pretrained_models/` folder.

```
└── root
    ...
    └── pretrained_models
        └── encoder_render.pt
        └── ffhqrebalanced512-128.pth
        └── model_ir_se50.pth        
```

## Usage

```bash
python inference.py --input ./imgs/input.png --output ./imgs/output.png --checkpoint ./checkpoints/ckpt.pth 
```

Notice that the input image should be preprocessed folowing the [EG3D](https://github.com/NVlabs/eg3d).
Note that the LT model implement has some difference with the original implement, but achieve the nearly same performance.

## TODO
- [x] Upload inference code and checkpoint.
- [x] Release Training code.
- [x] Release LT model and checkpoint.



## Acknowledgement
This repo is based on [triplanenet](https://github.com/anantarb/triplanenet), [EG3D](https://github.com/NVlabs/eg3d). Thanks to their great work!

```
@inproceedings{trevithick2023,
  author = {Alex Trevithick and Matthew Chan and Michael Stengel and Eric R. Chan and Chao Liu and Zhiding Yu and Sameh Khamis and Manmohan Chandraker and Ravi Ramamoorthi and Koki Nagano},
  title = {Real-Time Radiance Fields for Single-Image Portrait View Synthesis},
  booktitle = {ACM Transactions on Graphics (SIGGRAPH)},
  year = {2023}
}
```
```
@inproceedings{Chan2022,
  author = {Eric R. Chan and Connor Z. Lin and Matthew A. Chan and Koki Nagano and Boxiao Pan and Shalini De Mello and Orazio Gallo and Leonidas Guibas and Jonathan Tremblay and Sameh Khamis and Tero Karras and Gordon Wetzstein},
  title = {Efficient Geometry-aware {3D} Generative Adversarial Networks},
  booktitle = {CVPR},
  year = {2022}
}
```

```
@article{bhattarai2024triplanenet,
  title={TriPlaneNet: An Encoder for EG3D Inversion},
  author={Bhattarai, Ananta R. and Nie{\ss}ner, Matthias and Sevastopolsky, Artem},
  booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2024}
}
```
