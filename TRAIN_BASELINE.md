# Train

This document assumes that the instructions in `INSTALL.md` have been successfully executed, in particular the Installation Validation section.

## Data

### MOT Data

Download and unpack MOT datasets and CrowdHuman datasets that are important for finetuning in the `data` directory:

First, set `cd data`

1. [MOT17](https://motchallenge.net/data/MOT17/):

    ```
    # probably alrady done during installation validation
    wget https://motchallenge.net/data/MOT17.zip
    unzip MOT17.zip
    cd ..
    python src/generate_coco_from_mot.py
    ```

2. [MOT20](https://motchallenge.net/data/MOT20/):

    ```
    wget https://motchallenge.net/data/MOT20.zip
    unzip MOT20.zip
    cd ..
    python src/generate_coco_from_mot.py --mot20
    ```

3. [CrowdHuman](https://www.crowdhuman.org/download.html):

    1. Create a `CrowdHuman` and `CrowdHuman/annotations` directory.
    2. Download and extract the `train` and `val` datasets including their corresponding `*.odgt` annotation file into the `CrowdHuman` directory.
       I recommend to use the [gdown package](https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive) with the Google Drive ID.
    3. Create a `CrowdHuman/train_val` directory and merge or symlink the `train` and `val` image folders.
    4. Copy the contents of `annotation_traind.odgt` and `annotation-vald.odgt` into a file named `train_val.json`. A Json viewer helps you avoiding formatting issues when copying.
    4. Run `python src/generate_coco_from_crowdhuman.py`
    5. The final folder structure should resemble this:
        ~~~
        |-- data
            |-- CrowdHuman
            |   |-- train
            |   |   |-- *.jpg
            |   |-- val
            |   |   |-- *.jpg
            |   |-- train_val
            |   |   |-- *.jpg
            |   |-- annotations
            |   |   |-- annotation_train.odgt
            |   |   |-- annotation_val.odgt
            |   |   |-- train_val.json
        ~~~

### MCMOT Data: WILDTRACK

Create WILDTRACK dataset in COCO single-camera tracking format with

    cd data  
    wget http://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/Wildtrack/Wildtrack_dataset_full.zip  
    jar xf Wiltrack_dataset_full.zip  // unpacking without zip-bomb error
    cd ..  
    python src/wildtrack_generate_coco.py

Thereafter, **also** generate the same data in MOT format. This ugly double structure is necessary because
the evaluation code requires validation and test splits in MOT format. This fact is not written in the section of trackformer's `TRAINING.md` about training on a custom dataset. Yet, it is revealed in [this issue](https://github.com/timmeinhardt/trackformer/issues/73). Note that we only require the validation data but the following script convert also the training data.

    python src/wildtrack_generate_mot_from_coco.py
    # test whether COCO and MOT data are equal
    python wildtrack_test.py

## Pretrained Models

Download and unpack pretrained TrackFormer model files in the `models` directory:

    wget https://vision.in.tum.de/webshare/u/meinhard/trackformer_models_v1.zip
    unzip trackformer_models_v1.zip

## Finding a free GPU

Check which GPU, if any, is free with `$ nvidia-smi`. Select specific GPUS by prepending `CUDA_VISIBLE_DEVICES=0,1` before `python myscript.py`. Additionally, we sometimes have to empty the GPU cache 

    import gc  
    import torch  
    gc.collect()  
    torch.cuda.empty_cache()

To find students that potentially overuse their GPU limit, check the [MONICA GPU tab](http://monica.informatik.uni-mannheim.de/gpu/dws-student-01) that has a 5 minutes delay or use `$ ps aux | grep <PID>` with the PID from `$ nvidia-smi`.


## Monitoring

We use visdom to monitor the training process in terms of, for instance, training loss and evaluation metrics on the test split.

In `cfgs/train.yaml`, set `no_vis=false`, `vis_server: http://localhost`, and ` local port to, e.g. 8090` to activate the monitoring.

Before starting the training scripts, type the following in one terminal with activated environment to fire up the local server in the browser:

- `python -m visdom.server -p 8090`

Use the local address `127.0.0.1:8090` to access the page with the dashboard and click on the environment that we will define below.

## Training

The multiple GPU part contains the command for generating the baseline results. The single GPU part is for testing, developing and debugging.

### Training on single GPU

Follow up by opening another new terminal with activated environment and start the training process on a single GPU e.g. via

    CUDA_VISIBLE_DEVICES=0 python src/train.py with \
        wildtrack_mot_crowdhuman \
        deformable \
        multi_frame \
        tracking \
        output_dir=models/test_wildtrack_only

Switch to the browser window and change environment to the name of the output directory from the previous command e.g. `models/mot17_deformable_multi_frame`

### Training on multiple GPUs

Again, open another new terminal with activated environment. Type

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env src/train.py with \
        wildtrack_mot_crowdhuman \
        deformable \
        multi_frame \
        tracking \
        output_dir=models/baseline_wildtrack_mot_crowdhuman


## Process management

- Stard tmux with `tmux` and open visdom or start training to prevent process from detaching during system suspension or connection timeout (especially during VPN session).

- Use `kill -9 <PID>` to kill own detached processes 

