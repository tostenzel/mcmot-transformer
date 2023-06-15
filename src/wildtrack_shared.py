"""Visualize bounding boxes from COCO data.

Used to check COCO data in `generate_coco_from_wildtrack.py` and to check MOT
data from a back conversion to COCO format in `wildtrack_check_mot.py`

"""
import os
from typing import List, Optional

from PIL import Image
from pycocotools.coco import COCO
import skimage.io as io
from matplotlib import pyplot as plt

import wildtrack_globals as glob


DEST_COCO_ANNOTATIONS = f"{glob.ROOT}/annotations"


def check_coco_from_wildtrack(
        img_dir_path: str = f"{glob.ROOT}/train",
        coco_annotations_path: str = f"{DEST_COCO_ANNOTATIONS}/train.json",
        split: str = "train",
        write_path = "data/WILDTRACK/debug_coco_images",
        read_symlinked_symlinked_jpgs: bool = False,
        multicam=False,
        num_img = 5
    ) -> None:
    """
    Visualizes and stores generated COCO data. Only used for debugging.

    We save `num_img` files for each camera in data/WILDTRACK/debug_images.
    `validation_data` flag has to be true if we pass the validation data
    directories.

    Args:
        img_dir_path: path to images in .jpg format
        coco_annotations_path: path to COCO annotations with boxes that point to
            the images.
        split: "train", "test, or "val".
        write_path: path to where the images with bboxes are saved.
        read_symlinked_symlinked_jpgs: flag that shows whether symlinks of
            symlinks of jpgs are used for checking the data in MOT format.
        num_img: number of the first images in the folder that are saved with
            bounding boxes.

    """
    if os.path.isdir(write_path) is False:
        os.makedirs(write_path)
    
    # used for constructing a mapping between image and object annotations.
    coco = COCO(coco_annotations_path)
    cat_ids = coco.getCatIds(catNms=['person'])

    if multicam is False:
        n_cams = glob.N_CAMS
        # check the correctness of all image ids at once
        # img_ids = coco.getImgIds(catIds=cat_ids)
        #val_img_ids_offset = int((1 - TRAIN_SPLIT) * SEQ_LENGTH)
        if split == "train":
            img_id_offset = glob.TRAIN_SEQ_LENGTH
        elif split == "test":
            img_id_offset = glob.TEST_SEQ_LENGTH
        elif split == "val":
            img_id_offset = glob.VAL_SEQ_LENGTH
    else:
        n_cams = 1
        img_id_offset = 0

    for c in range(0, n_cams):

        for img_id in range(0, num_img):

            img_id = img_id + c * img_id_offset
            img_annotation = coco.loadImgs(img_id)[0]
            if read_symlinked_symlinked_jpgs is False:
                i = io.imread(img_dir_path + "/" + img_annotation['file_name'])
            else:
                i = io.imread(os.readlink(os.readlink(
                img_dir_path + "/" + img_annotation['file_name'])
            ))
            plt.imshow(i)
            plt.axis('off')
            ann_ids = coco.getAnnIds(
                imgIds=img_annotation['id'],
                catIds=cat_ids,
                iscrowd=None
                )
            anns = coco.loadAnns(ann_ids)
            coco.showAnns(anns, draw_bbox=True)
            plt.savefig(f'{write_path}/debug_{img_annotation["file_name"]}')
            # clear figures/bboxes for next picture
            plt.clf()


def validate_jpgs(multicam: bool=False):
    """
    Validate converted .jpgs.

    Sometimes some files were not valid and caused errors during trainig.
    Code according to
    https://stackoverflow.com/questions/46854496/python-script-to-detect-broken-images
    """
    if multicam is True:
        for id_ in glob.SEQUENCE_IDS:
            _val(id_)
    else:
        _val()

def _val(id_: Optional[str]=None):
    for split in ["train", "test", "val"]:
        if id_ is not None:
            path = f"{glob.MULTICAM_ROOT}/{id_}/{split}"
        else:
            path = f"{glob.ROOT}/{split}"
        for filename in os.listdir(path):
            if filename.endswith(".jpg"):
                try:
                    im = Image.open(f"{path}/{filename}")
                    #I perform also verify, don't know
                    # if he sees other types of defects
                    im.verify()
                    im.close() #reload is necessary in my case
                    im = Image.open(f"{path}/{filename}")
                    im.transpose(Image.FLIP_LEFT_RIGHT)
                    im.close()
                except (IOError, SyntaxError) as e:
                    print(e, filename)


COCO_BASE_DICT = {
    "info": {
        "year": 2021,
        "version": 1,
        "description": "WildTrack dataset",
        "contributor": "",
        "url": "https://www.epfl.ch/labs/cvlab/data/data-wildtrack/",
        "date_created": "2021-08-27"
    },
    "licenses": [{
        "id": 1,
        "name": "GPL 2",
        "url": "https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html"
    }],
    "images": [],
    "annotations": [],
    "categories": [
        {
            "id": 1,
            "name": "person",
            "supercategory": "person"
        }
    ],
    # tracking specific
    "frame_range": {"start": 0.0, "end": 1.0},
    "sequences": None
}


def flatten_listoflists(ll: List[list]) -> list:
    l = []
    for sublist in ll:
        l.extend(sublist)
    return l