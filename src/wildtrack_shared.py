"""Visualize bounding boxes from COCO data.

Used to check COCO data in `generate_coco_from_wildtrack.py` and to check MOT
data from a back conversion to COCO format in `wildtrack_check_mot.py`

"""
import os
from typing import List, Optional
from copy import deepcopy

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import skimage.io as io
from matplotlib import pyplot as plt

import wildtrack_globals as glob

from multicam_wildtrack_load_calibration import load_all_intrinsics
from multicam_wildtrack_load_calibration import load_all_extrinsics


# load list of rvec and tvecs and camera matrices
rvecs, tvecs = load_all_extrinsics()
camera_matrices, dist_coeffs = load_all_intrinsics()


DEST_COCO_ANNOTATIONS = f"{glob.ROOT}/annotations"


def convert_wildtrack_to_coco_bbox(xmax, xmin, ymax, ymin):
    x = xmin
    y = ymin
    w_box = xmax - xmin
    h_box = ymax - ymin
    return x, y, w_box, h_box


def check_coco_from_wildtrack(
        img_dir_path: str = f"{glob.ROOT}/train",
        coco_annotations_path: str = f"{DEST_COCO_ANNOTATIONS}/train.json",
        split: str = "train",
        write_path = "data/WILDTRACK/debug_coco_images",
        read_symlinked_symlinked_jpgs: bool = False,
        multicam=False,
        three_dim=False,
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

    # FIXME: TOBIAS changed from range(n_cams)
    for c in range(n_cams):

        seq_name = img_dir_path.rsplit("/")[-2]
        view_number = int(seq_name[-1])
        for img_id in range(0, num_img):

            img_id = img_id + c * img_id_offset
            img_annotation = coco.loadImgs(img_id)[0]

            if multicam is True:
                # we use the first annotation file with paths from c0 independent of real camera view...
                file_name = f"{seq_name}-{img_annotation['file_name'].rsplit('-')[1]}"
            else:
                file_name = img_annotation['file_name']

            if read_symlinked_symlinked_jpgs is False:
                    i = io.imread(img_dir_path + "/" + file_name)
            else:
                i = io.imread(os.readlink(os.readlink(
                img_dir_path + "/" + file_name)
            ))
            plt.imshow(i)
            plt.axis('off')
            ann_ids = coco.getAnnIds(
                imgIds=img_annotation['id'],
                catIds=cat_ids,
                iscrowd=None
                )
            cylinder_anns = coco.loadAnns(ann_ids)
            if three_dim is True:
                from multicam_wildtrack_3D_cylinder_to_2D_bbox_projections import transform_3D_cylinder_to_2D_bbox_params as get_bbox
                anns = []
                for cyl_ann in cylinder_anns:
                    bbox_annotation = deepcopy(cyl_ann)

                    cyl = {
                        "x_center": cyl_ann["bbox"][0],
                        "y_center": cyl_ann["bbox"][1],
                        "height": cyl_ann["bbox"][2],
                        "radius": cyl_ann["bbox"][3]
                    }
                    bbox = get_bbox(
                        cyl,
                        rvecs[view_number],
                        tvecs[view_number],
                        camera_matrices[view_number],
                        dist_coeffs[view_number]
                    )
                    if bbox is not None:
                        from wildtrack_shared import convert_wildtrack_to_coco_bbox
                        x, y, w_box, h_box = convert_wildtrack_to_coco_bbox(
                            bbox["xmax"],
                            bbox["xmin"],
                            bbox["ymax"],
                            bbox["ymin"]
                        )
                        bbox_arr = np.float32([x, y, w_box, h_box])
                        bbox_annotation["bbox"] = bbox_arr
                        anns.append(bbox_annotation)

            coco.showAnns(anns, draw_bbox=True)
            plt.savefig(f'{write_path}/debug_{file_name}')
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