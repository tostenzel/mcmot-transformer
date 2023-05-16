# Train

This document assumes that the instructions in `INSTALL.md` have been successfully executed, in particular the Installation Validation section.

## Data

### MOT Data

Download and unpack MOT datasets in the `data` directory:

First, set `cd data`

1. [MOT17](https://motchallenge.net/data/MOT17/):

    ```
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
    4. Unsure about this step: Also merge the annotations to `train_val.json`. Json is require by next step, therefore, this is my guess.
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

Thereafter, **also** generate the same data in MOT format. This ugly doubling is necessary because
the evaluation code requires validation/test data in MOT format. This fact is not written in the section of trackformer's `TRAINING.md` about training on a custom dataset. Yet, it is revealed in [this issue](https://github.com/timmeinhardt/trackformer/issues/73). Note that we only require the validation data but the following script convert also the training data. To call the python file, type

`python src/wildtrack_generate_mot_from_coco.py`


## Finding a free GPU

**Caution:** I am unsure about the following advices:

Check which GPU, if any, is free with `$ nvidia-smi`. If you only want to train on one GPU, set device: cuda:x with x in {0,...,7}. For more GPUs, use `export CUDA_VISIBLE_DEVICES=6,7` and afterwards `python myscript.py`.

    import gc  
    import torch  
    gc.collect()  
    torch.cuda.empty_cache()

To find students that potentially overuse their GPU limit, check the [MONICA GPU tab](http://monica.informatik.uni-mannheim.de/gpu/dws-student-01) that has a 5 minutes delay or use `$ ps aux | grep <PID>` with the PID from `$ nvidia-smi`.


## Monitoring

First, install visdom into the activated environment with `pip install visdom`.

In `cfgs/train.yaml`, set `no_vis=false`, `vis_server: http://localhost`, and ` local port to, e.g. 8090`

Adapt these settings in `cfgs/train.yaml`.

Type the following in one terminal with activated environment to fire up the local server and to open the monitoring window with the dynamic plot of the training metrics per iteration IN in the browser:

- `python -m visdom.server -p 8090`

Use the local address `127.0.0.1:8090` to access the page with the dashboard and click on the environment that we will define below.

Follow up by opening another new terminal with activated environment and start the training process e.g. via

    CUDA_VISIBLE_DEVICES=6 python src/train.py with \
        wildtrack_only \
        deformable \
        multi_frame \
        tracking \
        output_dir=models/test_wildtrack_only

Switch to the browser window and change environment to the name of the output directory from the previous command e.g. `models/mot17_deformable_multi_frame`

## Training on multiple GPUs

E.g. on MOT17 data from scratch

    NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env src/train.py with \
        wildtrack_only \
        deformable \
        multi_frame \
        tracking \
        output_dir=models/test_wildtrack_only

## Training with custom data

Fine-tune the MOT17 model on WILDTRACK data with

    NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=1,2 python src/train.py with \
        wildtrack_only \
        deformable \
        multi_frame \
        tracking \
        output_dir=models/test_wildtrack_only

## Process management

- Use `kill -9 PID` to kill own detached processes 
- Stard tmux with `tmux`

