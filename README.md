# McFormer: Multi-Camera Multi-object Tracking with Transformers

This repository hosts the implementation of the master's thesis "Multi-Camera Multi-object Tracking with Transformers" written by Tobias Stenzel and supervised by Paul Swoboda.

The codebase builds upon [DETR](https://github.com/facebookresearch/detr), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), [Tracktor](https://github.com/phil-bergmann/tracking_wo_bnw), and [Trackformer](https://github.com/timmeinhardt/trackformer).

## Branches

 - The default branch `24-switch-mcmot-to-detr` containts the final MCMOT model
 - Branch `1-train-trackformer-on-wildtrack` contains the single-camera model and the results on WILDTRACK
 - Branch `22-xminw-yminh-ww-hh-format-with-normal-detr` contains a sanity check for the MCMOT because it substitutes the transformation from 2D to 3D and back by predicting bounding box form (xmin/W, ymin/H, w/W, h/H) instead (xcenter/W, ycenter/H, w/W, h/H). It also contains the feature to set the learning rates for each transformer layer separately.

## Text

The thesis is accessible only for invitees in this [Overleaf project](https://www.overleaf.com/project/63c56d54d8857f5782b3d039).

## Installation

The [INSTALL.md](/INSTALL.md) describes the installation process on the server with the address `dws-student-01.informatik.uni-mannheim.de`.

## Training

The [TRAIN_BASELINE.md](/TRAIN_BASELINE.md) describes how to prepare and execute the training of the baseline model on WILDTRACK.
