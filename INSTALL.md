# How to Install McFormer on the Student Server

## Server connection

After registering for the student server, connect to it via
- `ssh <username>@dws-student-01.informatik.uni-mannheim.de`

## Conda environment manager and Python 3.7

We do not work in the `home` directory because it has only 50GB memory. Instead, we work in the `work` directory:

Switch to `<username>/work/` and install miniconda with Python 3.7 via
- `wget https://repo.anaconda.com/miniconda/Miniconda3-py37_23.1.0-1-Linux-x86_64.sh`  
- `chmod 755`  
- `./Miniconda3-py37_23.1.0-1-Linux-x86_64.sh`

## Clone project

- `git clone https://github.com/tostenzel/mcformer`  
- `cd mcformer`

## Conda Environment with Python 3.7

- The student server has 8 NVIDIA RTX A6000 GPUs (see `nvidea-smi`). According to the NVIDIA recommendation, we have to use at least CUDA 11.1 (see [CUDA wiki](https://en.m.wikipedia.org/wiki/CUDA#GPUs_supported) and [NVIDIA forum](https://forums.developer.nvidia.com/t/rtx-a6000-cuda-compatibility/218450)).

- The driver is NVIDIA UNIX x86_64 Kernel Module  525.89 (see `cat /proc/driver/nvidia/version`).

- We create the conda environment with Python 3.7 via
    - `conda create --prefix=.env/conda-py3_7 python=3.7 pip`
- and activate it immediately with
    - `conda activate .env/conda-py3_7`

## PyTorch

- For CUDA 11.1 (requirement from GPU), we have to choose at least PyTorch 1.8 (see [PyTorch versions](https://pytorch.org/get-started/previous-versions/)). However, the [Trackformer program](https://github.com/timmeinhardt/trackformer/), our main dependency, strongly recommends PyTorch 1.7 and torchvision 0.8 (see [this Issue](https://github.com/timmeinhardt/trackformer/issues/41)). Note that these two versions are unequal to what is incorrectly stated in Trackformer's [Install.md](https://github.com/timmeinhardt/trackformer/blob/main/docs/INSTALL.md). Because of Trackformer, I choose Pytorch 1.7.1 that has only CUDA 11.0 instead of 11.1 support (see [PyTorch versions](https://pytorch.org/get-started/previous-versions/)) and hope that CUDA works despite it.

- `conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch`

Unfortunatley, if we install the cudatoolkit via conda, the `nvcc compiler` does not come with it. Therefore, we need to install it on top manually to prevent CUDA from choosing the inappropropriate, locally pre-installed version in `/usr/local/cuda`. Unfavorably, there is no compiler for our CUDA 11.0 (that we have chosen for PyTorch 1.7.1). Therefore, I choose a higher version one more time:

 - `conda install -c "nvidia/label/cuda-11.7.0" cuda-nvcc`  
 - `conda install -c conda-forge cudatoolkit-dev`

 ## MultiScaleDeformableAttention

Trackformer's main dependency is MultiScaleDeformableAttention from the [Deformable-DETR repository](https://github.com/fundamentalvision/Deformable-DETR). That program is used for detecting images (not videos) as a more efficient DETR version. Installing this program is tricky. Thus, we do it before the other Trackformer requirements.

 We first install pycocotools (with fixed ignore flag) according to the Trackformer installation guide. That dependency is also used by Deformable-DETR without the version specification.

- `pip3 install -U 'git+https://github.com/timmeinhardt/cocoapi.git#subdirectory=PythonAPI'`

We continue with the Deformable-DETR requirements:

- `pip3 install -r requirements-deformable_detr.txt`

We follow up with our requirements (only tmux):

- `pip3 install -r requirements-mcmot-transformer.txt`

Next, we install MultiScaleDeformableAttention from the local files in this repository with

- `python src/trackformer/models/ops/setup.py build --build-base=src/trackformer/models/ops/ install`

Finally, we test whether the installation was succesful:

    cd src/trackformer/models/ops
    # unit test (should see all checking is True)
    python test.py
    cd ../../../..

## Trackformer

At last, we install the [other Trackformer requirements](https://github.com/timmeinhardt/trackformer/blob/main/requirements.txt). Note that I have changed numpy to a more recent numpy version. Without that, we would get an DimensionMismatchError from running Trackformer's `src/track.py`.

- `pip3 install -r requirements-trackformer.txt`

We test whether the Trackformer installation was successful in two ways:

## Installation Validation

#### 1. Evaluation

Either download the MOT17(https://motchallenge.net/data/MOT17/) dataset to the data folder via 

-     cd data
      wget https://motchallenge.net/data/MOT17.zip
      jar xf MOT17.zip   # unzip might yield possible zip bomb error
      python src/generate_coco_from_mot.py

Then, download and unpack the pretrained TrackFormer model files in the models directory:

-     cd models
      wget https://vision.in.tum.de/webshare/u/meinhard/trackformer_models_v1.zip
      jar xf trackformer_models_v1.zip   # unzip might yield possible zip bomb error
      cd ..

Next, evaluate the pre-trained MOT17 models with MOT20 metrics via
- `python src/track.py`

#### 2. Training

Try to train Trackformer on the MOT17 dataset for some batches via

-     python src/train.py with \
          mot17 \
          deformable \
          multi_frame \
          tracking \
          output_dir=models/mot17_deformable_multi_frame \
