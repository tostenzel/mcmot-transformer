"""Visualize bounding boxes from COCO data.

Used to check COCO data in `generate_coco_from_wildtrack.py` and to check MOT
data from a back conversion to COCO format in `wildtrack_check_mot.py`

"""
import os

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
        num_img = 20
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
    # check the correctness of all image ids at once
    # img_ids = coco.getImgIds(catIds=cat_ids)
    #val_img_ids_offset = int((1 - TRAIN_SPLIT) * SEQ_LENGTH)
    if split == "train":
        img_id_offset = glob.TRAIN_SEQ_LENGTH
    elif split == "test":
        img_id_offset = glob.TEST_SEQ_LENGTH
    elif split == "val":
        img_id_offset = glob.VAL_SEQ_LENGTH
       
    for c in range(0, glob.N_CAMS):

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
