# Unofficial implement of Live3DPortrait
This is a unofficial implement of [Live3DPortrait: Real-Time Radiance Fields for Single-Image Portrait View Synthesis](https://research.nvidia.com/labs/nxp/lp3d/) in Pytorch. 

![display1](./docs/display1.png)
![display2](./docs/display2.png)
![display3](./docs/display3.png)

<video width="320" height="240" controls>
    <source src="./docs/output.mp4" type="video/mp4">
</video>


## Pretrained Model
The checkpoints can be download from [here](https://drive.google.com/drive/folders/1Z6uri8pH048Qzyhu0PE3llxO5EC7ic_v?usp=drive_link). 
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

## TODO
- [x] Upload inference code and checkpoint.
- [ ] Release Training code.
- [ ] Release LT model and checkpoint.




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
