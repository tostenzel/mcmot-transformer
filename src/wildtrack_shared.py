"""Functions and data used in multiple `multicam_wildtrack_...` and
`wildtrack_...` modules.

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
from multicam_wildtrack_3D_cylinder_to_2D_bbox_projections import transform_3D_cylinder_to_2D_bbox_params as get_bbox


# load list of rvec and tvecs and camera matrices
rvecs, tvecs = load_all_extrinsics()
camera_matrices, dist_coeffs = load_all_intrinsics()


DEST_COCO_ANNOTATIONS = f"{glob.ROOT}/annotations"


def convert_wildtrack_to_coco_bbox(xmax, xmin, ymax, ymin):
    """Converts the bbox format used in WILDTRACK to COCO.
    
    Both assume that the origin of an image is the upper left pixel.
    The x and y coordinates for coco represent the upper left bbox corner.
    
    """
    x = xmin
    y = ymin
    width = xmax - xmin
    height_box = ymax - ymin
    return x, y, width, height_box


def check_coco_from_wildtrack(
        three_dim_multicam=False,
        img_dir_path: str = f"{glob.ROOT}/train",
        coco_annotations_path: str = f"{DEST_COCO_ANNOTATIONS}/train.json",
        split: str = "train",
        write_path = "data/WILDTRACK/debug_coco_images",
        read_symlinked_symlinked_jpgs: bool = False,
        num_img = 5
    ) -> None:
    """Visualizes and stores generated COCO data.

    Used to check COCO data in `wildtrack_generate_coco.py`,
    multicam_wildtrack_generate_coco.py`, and to check MOT
    data from a back conversion to COCO format in `wildtrack_check_mot.py`

    Args:
        three_dim_multicam: Whether are 3D multicamera dataset should be checked.
            There, we only have annotations for the first sequence `c0` that
            are used to get bbox for images from all other views from different
            folders.
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
    # 'data/multicam_WILDTRACK/c1/train' gives 1
    if three_dim_multicam is True:
        view_number = int(img_dir_path.split('/')[-2][1])

    if os.path.isdir(write_path) is False:
        os.makedirs(write_path)
    
    # used for constructing a mapping between image and object annotations.
    coco = COCO(coco_annotations_path)
    cat_ids = coco.getCatIds(catNms=['person'])

    if three_dim_multicam is True:
        n_cams = 1
        img_id_offset = 0
    else:
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

    for c in range(n_cams):

        seq_name = glob.SEQUENCE_IDS[c]
        for img_id in range(0, num_img):

            img_id = img_id + c * img_id_offset
            img_annotation = coco.loadImgs(img_id)[0]

            if three_dim_multicam is True:
                # we use the first annotation file with paths from c0 for all camera views...
                file_name = f"{seq_name}-{img_annotation['file_name'].rsplit('-')[1]}"
                # handle that we write annotations only for first sequence
                # so that the image name is wrong for other sequences:
                # e.g. 'c0-00000000.jpg' is used for 'c1-00000000.jpg'.
                file_name = file_name.replace(
                    "c0",f"{glob.SEQUENCE_IDS[view_number]}"
                )
            else:
                file_name = img_annotation["file_name"]

            img_path = img_dir_path + "/" + file_name
            if read_symlinked_symlinked_jpgs is False:
                i = io.imread(img_path)
            else:
                i = io.imread(os.readlink(os.readlink(img_path)
            ))
            plt.imshow(i)
            plt.axis("off")
            ann_ids = coco.getAnnIds(
                imgIds=img_annotation["id"],
                catIds=cat_ids,
                iscrowd=None
                )
            anns = coco.loadAnns(ann_ids)
            if three_dim_multicam is True:
                cylinder_anns = []
                for cyl_ann in anns:
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
                        x, y, w_box, h_box = convert_wildtrack_to_coco_bbox(
                            bbox["xmax"],
                            bbox["xmin"],
                            bbox["ymax"],
                            bbox["ymin"]
                        )
                        bbox_arr = np.float32([x, y, w_box, h_box])
                        bbox_annotation["bbox"] = bbox_arr
                        # If the area field is not field by the user,
                        # the bboxes are not shown correctly.
                        bbox_annotation["area"] = w_box * h_box
                        cylinder_anns.append(bbox_annotation)
                anns = cylinder_anns

            coco.showAnns(anns, draw_bbox=True)
            plt.savefig(f"{write_path}/debug_{file_name}")
            # clear figures/bboxes for next picture
            plt.clf()



def validate_jpgs(multicam: bool=False):
    """Validate converted .jpgs.

    Sometimes some files were not valid and caused errors during trainig.
    Code according to
    https://stackoverflow.com/questions/46854496/python-script-to-detect-broken-images

    """
    if multicam is True:
        for id_ in glob.SEQUENCE_IDS:
            _validate_in_sequence_folder(id_)
    else:
        _validate_in_sequence_folder()


def _validate_in_sequence_folder(id_: Optional[str]=None):
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


def flatten_listoflists(ll: List[list]) -> list:
    l = []
    for sublist in ll:
        l.extend(sublist)
    return l


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
