"""Generates WILTRACK in MOT format from previously generated COCO files.

Although not stated in the section about training cumstom datasets in
trackformer's TRAIN.md,
[this Issue](https://github.com/timmeinhardt/trackformer/issues/73#issuecomment-1488421264)
reveals that trackformers evaluation code requires the training and validation
data in MOT format.
This script's output mimics `data/MOT17/test` for WILDTRACK without detections
(`det.txt`)
The module `wildtrack_sequence.py` contains the dataloader for one sequence
that is used in the evaluation.

Format
======
Definitions are backwards inferred from `generate_coco_from_mot.py`,
`mot17_sequence.py`, and https://motchallenge.net/instructions/
gt.txt (ground truth) -- has length 20202 for 13-FRCNN
<frame_number, track_id (sort key), x, y, w, h, class (person), class certainty,
    visibility>
ATTENTION: Starting indices may be either 0 or 1 and differ between formats.
Yet, this appears to make no differences for the ML functions.

"""
import json
import os
import copy

import numpy as np
import configparser
import skimage.io as io

import wildtrack_globals as glob

COCO_ROOT = glob.ROOT
SPLITS_SEQ_LENGTH = {"test": glob.TEST_SEQ_LENGTH, "val": glob.VAL_SEQ_LENGTH}
INI_DICT = {
    "name": "",
    "imDir":"img1",
    "frameRate": str(glob.ANNOTATED_FPS),
    "seqLength": "",
    "imWidth": str(glob.W),
    "imHeight": str(glob.H),
    "imExt":".jpg"
}
configparser = configparser.ConfigParser()
# no automatic conversion from capital letters to lower case
configparser.optionxform = str


def generate_mot_from_coco(coco_root: str=COCO_ROOT, multicam: bool=False) -> None:
    # We only need the validation data in MOT format for the evaluation code,
    # i.e. test and validation splits
    for split in ["test", "val"]:

        _create_mot_dirs(split, coco_root, multicam)

        file = open(f"{coco_root}/annotations/{split}.json")
        dataset_sequence = json.load(file)

        _create_img_symlinks(split, dataset_sequence, coco_root)
        _create_ground_truth(split, dataset_sequence, coco_root, multicam)
        _create_seqinfo_ini_files(split, coco_root, multicam)


def _create_mot_dirs(split, coco_root, multicam):
    """Create directory structure.


    For instance, for sequence "c0":
    |--mot-eval
        |--/c0-test
        |   |--/gt
        |   |   |--c0-test_gt.txt
        |   |--/img1 (containing symlinks to image files in COCO directory)
        |   |   |-- *.jpg
            |--seqinfo.ini
        
    """
    dir1 = f"{coco_root}/mot-eval"
    if os.path.isdir(dir1) is False:
        os.mkdir(dir1)
    if multicam is False:
        seq_ids = glob.SEQUENCE_IDS
    else:
        seq_ids = [glob.SEQUENCE_IDS[0]]

    for seq in seq_ids:
        dir2 = f"{coco_root}/mot-eval/{seq}-{split}"
        # folder for .det and .ini files
        if os.path.isdir(dir2) is False:
            os.mkdir(dir2)
        # folder for ground truth (MOT annotations)
        if os.path.isdir(f"{dir2}/gt") is False:
            os.mkdir(f"{dir2}/gt")
        # folder for symlinked img files
        if os.path.isdir(f"{dir2}/img1") is False:
            os.mkdir(f"{dir2}/img1")


def _create_img_symlinks(split, dataset_sequence, coco_root) -> None:
    """Create image files in folder `img1` via symlink to the COCO folder."""
    images = dataset_sequence["images"]

    # create symlinks for images
    for img in images:
        src = f"{coco_root}/{split}/{img['file_name']}"
        seq = img['file_name'].split('-')[0]
        dotjpg = img['file_name']
        dst = f"{coco_root}/mot-eval/{seq}-{split}/img1/{dotjpg}"
        #if os.path.isfile(dst) is True:
        #    continue
        #    #print(f"{dst} already exists. Do not write file.")
        #else:    
        os.symlink(src, dst)
        try:
            io.imread(os.readlink(dst))
        except:
            raise LookupError


def _create_ground_truth(split, dataset_sequence, coco_root, multicam: bool=False) -> None:
    """Create gt.txt files for each sequence.
    
    These files contain annotations and bounding boxes

    """
    annotations = dataset_sequence["annotations"]

    gt_file_dict = {}
    if multicam is False:
        seq_ids = glob.SEQUENCE_IDS
    else:
        seq_ids = [glob.SEQUENCE_IDS[0]]
    for seq in seq_ids:
        gt_file_dict[f"{seq}-{split}"] = []

    for ann in annotations:
        row = np.zeros([1, 9],dtype=int)
        image = dataset_sequence["images"][ann["image_id"]]
        if image["id"] != ann["image_id"]:
            # Then implement correct image selection with inefficient for loop.
            raise IndexError
        # <frame_number, track_id (sort key), x, y, w, h, class (person), class certainty,
        # visibility>
        # Attention: Add 1 b/c sequence class starts bbox dict indices at 1
        frame_id = image["frame_id"]
        track_id = ann["track_id"]
        x, y, w, h = ann["bbox"][0], ann["bbox"][1], ann["bbox"][2], ann["bbox"][3]
        obj_class = 1
        obj_certainty = 1
        visibility = ann["visibility"]
        row = [
            frame_id, track_id, x, y, w, h, obj_class, obj_certainty, visibility
        ]
        gt_file_dict[ann["seq"]].append(row)

    for k in gt_file_dict.keys():
        gt_file_dict[k] = np.vstack(gt_file_dict[k]).astype(int)

    for seq in seq_ids:
        # arr sorted by track id, then frame_id
        # see https://stackoverflow.com/questions/29352511/numpy-sort-ndarray-on-multiple-columns
        # pass col indices in reverse order
        sorted_arr = copy.deepcopy(gt_file_dict[f"{seq}-{split}"])
        sorted_arr = sorted_arr[np.lexsort((sorted_arr[:,0], sorted_arr[:,1]))]
        gt_file_dict[f"{seq}-{split}"] = sorted_arr
        dir2 = f"{coco_root}/mot-eval/{seq}-{split}"
        if os.path.isdir(dir2) is False:
            os.mkdir(dir2)
        # fmt="'%i'" for writing integers (avoiding many decimals for readability)
        np.savetxt(
            f"{dir2}/gt/{seq}-{split}_gt.txt",
            gt_file_dict[f"{seq}-{split}"],
            delimiter=",", fmt="%i"
        )


def _create_seqinfo_ini_files(split, coco_root, multicam: bool) -> None:
    """Create `seqinfo.ini`-files."""
    if multicam is False:
        seq_ids = glob.SEQUENCE_IDS
    else:
        seq_ids = [glob.SEQUENCE_IDS[0]]

    for seq in seq_ids:
        dir2 = f"{coco_root}/mot-eval/{seq}-{split}"

        # write .ini
        #https://docs.python.org/3/library/configparser.html
        configparser["Sequence"] = copy.deepcopy(INI_DICT)
        configparser["Sequence"]["name"] = f"{seq}-{split}"
        configparser["Sequence"]["seqLength"] = f"{SPLITS_SEQ_LENGTH[split]}"
        # save to a file
        with open(f"{dir2}/seqinfo.ini", "w") as configfile:
            configparser.write(configfile)


if __name__ == "__main__":
    generate_mot_from_coco()
