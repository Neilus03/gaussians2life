# Gaussians-to-Life: Text-Driven Animation of 3D Gaussian Splatting Scenes

### [Thomas Wimmer](https://wimmerth.github.io)<sup>1</sup>, [Michael Oechsle](https://moechsle.github.io/)<sup>2</sup>, [Michael Niemeyer](https://m-niemeyer.github.io/)<sup>2</sup>, [Federico Tombari](https://federicotombari.github.io/)<sup>1,2</sup>

<sup>1</sup>Technical University of Munich, <sup>2</sup>Google Research

Proceedings of the International Conference on 3D Vision (**3DV**), 2025

[**Project Website**](https://wimmerth.github.io/gaussians2life.html) | [PDF](https://arxiv.org/pdf/2411.19233)

---

### Abstract

State-of-the-art novel view synthesis methods achieve impressive results for multi-view captures of static 3D scenes.
However, the reconstructed scenes still lack “liveliness,” a key component for creating engaging 3D experiences.
Recently, novel video diffusion models generate realistic videos with complex motion and enable animations of 2D images,
however they cannot naively be used to animate 3D scenes as they lack multi-view consistency. To breathe life into the
static world, we propose _Gaussians2Life_, a method for animating parts of high-quality 3D scenes in a Gaussian
Splatting representation. Our key idea is to leverage powerful video diffusion models as the generative component of our
model and to combine these with a robust technique to lift 2D videos into meaningful 3D motion. We find that, in
contrast to prior work, this enables realistic animations of complex, pre-existing 3D scenes and further enables the
animation of a large variety of object classes, while related work is mostly focused on prior-based character animation,
or single 3D objects. Our model enables the creation of consistent, immersive 3D experiences for arbitrary scenes.

---

### Installation Instructions

As we used [threestudio](https://github.com/threestudio-project/threestudio) as framework, you will first need to setup
threestudio, which can be tricky at times.
In the following, we provide a step-by-step guide to install the necessary dependencies for this project in the hope
that it will help you to get started. We provide slightly modified requirements.txt files and recommend to use them
instead of threestudio's original requirements.txt file. You will need your
[CUDA Compute Capability](https://developer.nvidia.com/cuda-gpus) for the installation of tinycudann, a dependency of
threestudio.

```
git clone https://github.com/threestudio-project/threestudio.git
cd threestudio/custom/
git clone --recursive https://github.com/wimmerth/gaussians2life.git
mamba create -n gaussians2life python=3.10
mamba activate gaussians2life
cd gaussians2life
mamba install pytorch torchvision pytorch-cuda=[your CUDA version, e.g., 11.8] cudnn xformers -c pytorch -c nvidia -c xformers
export TORCH_CUDA_ARCH_LIST="[your CUDA compute capability, e.g., 8.9]"
export TCNN_CUDA_ARCHITECTURES="[your CUDA compute capability, e.g., 89]"
pip install --upgrade pip setuptools ninja
pip install git+https://github.com/KAIR-BAIR/nerfacc.git@v0.5.2
pip install git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch
pip install -r requirements.txt
pip install ./diff-gaussian-rasterization
pip install ./simple-knn
```

### Download of pretrained DynamiCrafter model

```
cd guidance/dynamicrafter
mkdir pretrained_models
cd pretrained_models
huggingface-cli download Doubiiu/DynamiCrafter_512 model.ckpt --local-dir ./
mv model.ckpt dynamicrafter-512.ckpt
```

All other models needed for this project are downloaded on the fly using huggingface.

### Running on Examples

We provide a few examples that are also shown in the paper. The respective 3DGS scenes and initially generated videos
can be found in `/assets/sample_data` and `/assets/sample_initial_videos`, respectively. You can run the following command
from the threestudio directory to reproduce the results, e.g., the bear scene.

```
gpu=0
python launch.py --config custom/gaussians2life/configs/bear.yaml --train --gpu $gpu system.prompt_processor.prompt="bear statue turns its head, static camera"
```