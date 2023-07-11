"""Generate `data/multicam_WILDTRACK´ (.jpg) from original
`data/Wildtrack_dataset` (.png) in sequence-specific folders with
sequence-specific annotations files (only first file used by model).

We only store 3D data for the train split, but 2D for test and val!!!

In contrast to the module `wildtrack_generate_3d_coco.py`,
we store 3D cylinder parameters generated from the information of all views
in the COCO bbox slot. We only do this for the first view/sequence `c0` and load
these annotations so that they are used for the joint input images from all
views when the configuration `three_dim_multicam: true` is set.

Still, we reuse the `_create_coco_files` function from
`wildtrack_generate_3d_coco.py` and the other functions are also very similar.

NOTE: I filled the COCO "area" field with -1. This might cause problems in
later in the training pipeline.

I use np.float32 because the cv2 functions require this type. The imprecisions
in the data most likely come from imprecisions in the camera calibrations.

"""
from typing import List

import copy
import json
import os

import numpy as np
import tqdm

import wildtrack_globals as glob
from multicam_wildtrack_load_calibration import load_all_intrinsics
from multicam_wildtrack_load_calibration import load_all_extrinsics
from multicam_wildtrack_2D_bbox_to_3D_cylinder_projections import \
    transform_2D_bbox_to_3D_cylinder_params as get_cylinder
from wildtrack_shared import convert_wildtrack_to_coco_bbox
from wildtrack_generate_coco import _create_coco_files
from wildtrack_shared import check_coco_from_wildtrack
from wildtrack_shared import validate_jpgs
from wildtrack_shared import COCO_BASE_DICT
from target_transforms import prevent_empty_bboxes


# Now destination paths inside cam folder
for id_ in glob.SEQUENCE_IDS:
    _dest_coco_annotations = f"{glob.MULTICAM_ROOT}/{id_}/annotations"
    if os.path.isdir(_dest_coco_annotations) is False:
        os.makedirs(_dest_coco_annotations)

# load list of rvec and tvecs and camera matrices
rvecs, tvecs = load_all_extrinsics()
camera_matrices, dist_coeffs = load_all_intrinsics()


def generate_3D_coco_from_wildtrack() -> None:
    """Create one single-camera tracking coco WILDTRACKdataset with seven seqs.

    """
    # each annotation file contains info for all cameras
    train_dataset = copy.deepcopy(COCO_BASE_DICT)
    train_dataset["sequences"] = [id + "-train" for id in glob.SEQUENCE_IDS]
    test_dataset = copy.deepcopy(COCO_BASE_DICT)
    test_dataset["sequences"] = [id + "-test" for id in glob.SEQUENCE_IDS]
    val_dataset = copy.deepcopy(COCO_BASE_DICT)
    val_dataset["sequences"] = [id + "-val" for id in glob.SEQUENCE_IDS]

    number_train_files = int(glob.TRAIN_SPLIT*len(glob.ANNOTATION_FILES))
    number_test_files = int(glob.TEST_SPLIT*len(glob.ANNOTATION_FILES))

    train_annotation_files = glob.ANNOTATION_FILES[
        :number_train_files
    ]
    test_annotation_files = glob.ANNOTATION_FILES[
        number_train_files:(number_train_files + number_test_files)
    ]
    val_annotation_files = glob.ANNOTATION_FILES[
        (number_train_files + number_test_files):
    ]

    # flexible annotation id for uneven annotation number per camera and sequence
    train_ann_id = 0
    test_ann_id = glob.TRAIN_SEQ_LENGTH
    val_ann_id = glob.TRAIN_SEQ_LENGTH + glob.TRAIN_SEQ_LENGTH

    output_train_annotation = "train.json"
    output_test_annotation = "test.json"
    output_val_annotation = "val.json"
    # analysis lists used to analyze how to best scale the cylinders to [0,1]^4
    for c in tqdm.tqdm(range(glob.N_CAMS)):

        train_images, train_annotations, train_ann_id, train_analysis_list = _create_3D_annotations(
            train_annotation_files, c, "train", train_ann_id
            )
        test_images, test_annotations, test_ann_id, test_analysis_list = _create_3D_annotations(
            test_annotation_files, c, "test", test_ann_id
            )
        val_images, val_annotations, val_ann_id, val_analysis_list = _create_3D_annotations(
            val_annotation_files, c, "val", val_ann_id
            )
        
        if c==0:
            # only train data used
            cylinder_analysis_array = np.vstack(train_analysis_list)
            os.makedirs(f"{glob.MULTICAM_ROOT}/cylinder_analysis", exist_ok=True)           
            np.save(
                f"{glob.MULTICAM_ROOT}/cylinder_analysis/cylinder_analysis_array.npy",
                cylinder_analysis_array
            )

        DEST_COCO_TRAIN = f"{glob.MULTICAM_ROOT}/{glob.SEQUENCE_IDS[c]}/train"
        DEST_COCO_TEST = f"{glob.MULTICAM_ROOT}/{glob.SEQUENCE_IDS[c]}/test"
        DEST_COCO_VAL = f"{glob.MULTICAM_ROOT}/{glob.SEQUENCE_IDS[c]}/val"

        for d in [DEST_COCO_TRAIN, DEST_COCO_TEST, DEST_COCO_VAL]:
            if os.path.isdir(d) is False:
                os.mkdir(d)

        # time consuming: I do not symlink previously generated *jpgs
        _create_coco_files(
            train_dataset,
            train_images,
            train_annotations,
            # _dest_coco_annotations
            f"{glob.MULTICAM_ROOT}/{glob.SEQUENCE_IDS[c]}/annotations/{output_train_annotation}",
            DEST_COCO_TRAIN
        )
        _create_coco_files(
            test_dataset,
            test_images,
            test_annotations,
            f"{glob.MULTICAM_ROOT}/{glob.SEQUENCE_IDS[c]}/annotations/{output_test_annotation}",
            DEST_COCO_TEST
        )
        _create_coco_files(
            val_dataset,
            val_images,
            val_annotations,
            f"{glob.MULTICAM_ROOT}/{glob.SEQUENCE_IDS[c]}/annotations/{output_val_annotation}",
            DEST_COCO_VAL
        )


def _create_3D_annotations(
        ann_files: List[dict],
        c: int,
        split: str="train",
        start_annotation_id: int = 0,
        create_analysis_list: bool = True
        ) -> tuple([List[dict], List[dict]]):
    """Creates annotations for every object on each image of a single-camera
    train, test or validation split.

    Here, the main difference is that we convert the 2D image bboxes to 3D
    cylinders using the mean cylinder params over all views where an object
    appears.

    Differences to `create_annotations` in `wildtrack_generate_coco.py`
    are documented in this function as comments.

    """
    if split == "train":
        seq_name_appendix = "-train"
        seq_length = glob.TRAIN_SEQ_LENGTH
    elif split =="test":
        seq_name_appendix = "-test"
        seq_length = glob.TEST_SEQ_LENGTH
    elif split == "val":
        seq_name_appendix = "-val"
        seq_length = glob.VAL_SEQ_LENGTH

    img_id = 0
    ann_id = start_annotation_id
    images = []
    annotations = []

    analysis_list = []
    #---------------------------------------------------------------------------
    # 3d coordinates only in annoation of c0, separate folders and anns per seq
    if split == "train":
        # loop length depends mainly on NUM_3D_HEIGHT_GRID_POINTS from
        # `multicam_wildtrack_2D_bbox_to_3D_cylinder_projections.py`
        for ann_file in ann_files:
            data = json.load(open(glob.SRC_ANNS + "/" + ann_file, "r"))  # was .json

            image_name = ann_file.rsplit(".", 1)[0] + ".jpg"
            image_name = f"c{c}-" + ann_file.rsplit(".", 1)[0] + ".jpg"
            images.append({
                "file_name": f"{image_name}",
                "height": glob.H,
                "width": glob.W,
                "id": img_id,# + img_id_offset,
                "license": 1,
                "frame_id": img_id,
                "seq_length": seq_length,
                "first_frame_image_id": 0# + img_id_offset
            })
            # generate object specific data (esp. 3D cylinders) only once.
            if c == 0:
                for instance in data:
                    cylinder_list = []
                    # always use max number of cameras because there may be some
                    # empty cylinder_arrs that the code does not handle for now
                    for cam in range(7):
                        # only use data for the selected camera and not for all others,
                        # where the same person is also visible
                        xmax, ymax, xmin, ymin = instance["views"][cam]["xmax"], instance["views"][cam]["ymax"], instance["views"][cam][
                        "xmin"], instance["views"][cam]["ymin"]
                        if not (xmax == -1 or ymax == -1 or xmin == -1 or ymin == 1):
                            #x = xmin
                            #y = ymin
                            #w_box = xmax - xmin
                            #h_box = ymax - ymin
                            cylinder = get_cylinder(instance["views"][cam], rvecs[cam], tvecs[cam], camera_matrices[cam])
                            cylinder_arr = np.fromiter(cylinder.values(), dtype=float)
                            cylinder_list.append(cylinder_arr)
                        
                    cylinder_mean = np.mean(cylinder_list, axis=0)
                    cylinder_list = []
                    annotations.append({
                        "id": ann_id,# + annotation_id_offset,
                        "bbox": [
                            # rounding not here but in proction from 3D to 2D.
                            cylinder_mean[0],
                            cylinder_mean[1],
                            cylinder_mean[2],
                            cylinder_mean[3]
                            ],
                        "image_id": img_id,# + img_id_offset
                        "segmentation": [],
                        "visibility": 1.0,
                        # "area": w_box * h_box,
                        "area": -1,
                        "category_id": 1,
                        "iscrowd": 0,
                        "seq": f"c{c}" + seq_name_appendix,
                        "track_id": instance["personID"]
                    })
                    if create_analysis_list is True:
                        analysis_list.append(
                            np.array([
                                cylinder_mean[0],
                                cylinder_mean[1],
                                cylinder_mean[2],
                                cylinder_mean[3]
                                ]
                            )
                        )
                    ann_id += 1
            img_id += 1
    #---------------------------------------------------------------------------
    # 2D coordinates in annotation of seq in separate folders
    # straight copy from `wildtrack_generate_coco.py`` except img_id_offset       
    else:
        for ann_file in ann_files:
            data = json.load(open(glob.SRC_ANNS + "/" + ann_file, "r"))  # was .json

            image_name = ann_file.rsplit(".", 1)[0] + ".jpg"
            image_name = f"c{c}-" + ann_file.rsplit(".", 1)[0] + ".jpg"
            images.append({
                "file_name": f"{image_name}",
                "height": glob.H,
                "width": glob.W,
                "id": img_id,# + img_id_offset,
                "license": 1,
                # tracking specific
                # `frame_id` is the img"s position relative to its sequence,
                # not the whole dataset (0 - 400),
                # see https://github.com/timmeinhardt/trackformer/issues/33#issuecomment-1105108004
                # Starts from 1 in MOT format
                "frame_id": img_id,
                "seq_length": seq_length,
                "first_frame_image_id": 0# + img_id_offset
            })

            for instance in data:
                # only use data for the selected camera and not for all others,
                # where the same person is also visible
                xmax, ymax, xmin, ymin = instance["views"][c]["xmax"], instance["views"][c]["ymax"], instance["views"][c][
                "xmin"], instance["views"][c]["ymin"]
                if not (xmax == -1 or ymax == -1 or xmin == -1 or ymin == 1):
                    x, y, w_box, h_box = convert_wildtrack_to_coco_bbox(xmax, xmin, ymax, ymin)
                    annotations.append({
                        "id": ann_id,# + annotation_id_offset,
                        "bbox": [
                            int(round(x)),
                            int(round(y)),
                            int(round(w_box)),
                            int(round(h_box))
                            ],
                        "image_id": img_id,# + img_id_offset, #+ val_img_id_offset,
                        "segmentation": [],
                        #"ignore":,
                        "visibility": 1.0,
                        "area": w_box * h_box,
                        "category_id": 1,
                        "iscrowd": 0,
                        # tracking specific
                        "seq": f"c{c}" + seq_name_appendix,
                        # TODO: perhaps offset, too? Yet, this info should make baseline stronger.
                        "track_id": instance["personID"]
                    })

                ann_id += 1
            img_id += 1
    #---------------------------------------------------------------------------
    return images, annotations, ann_id, analysis_list


if __name__ == "__main__":
    generate_3D_coco_from_wildtrack()

    # annotation path must be fixed to c0 and converted to sequence in check
    # to load the right image?
    for id_ in glob.SEQUENCE_IDS:
        check_coco_from_wildtrack(
            three_dim_multicam=True,
            img_dir_path = f"{glob.MULTICAM_ROOT}/{id_}/train",
            write_path = f"{glob.MULTICAM_ROOT}/debug_coco_images",
            # Fix to c0
            coco_annotations_path = f"{glob.MULTICAM_ROOT}/c0/annotations/train.json",
            num_img=5
        )

        check_coco_from_wildtrack(
            three_dim_multicam=False,
            img_dir_path = f"{glob.MULTICAM_ROOT}/{id_}/val",
            write_path = f"{glob.MULTICAM_ROOT}/debug_coco_images",
            # Fix to c0
            coco_annotations_path = f"{glob.MULTICAM_ROOT}/{id_}/annotations/val.json",
            num_img=5,
            no_img_id_offset=True
        )

        check_coco_from_wildtrack(
            three_dim_multicam=False,
            img_dir_path = f"{glob.MULTICAM_ROOT}/{id_}/test",
            write_path = f"{glob.MULTICAM_ROOT}/debug_coco_images",
            # Fix to c0
            coco_annotations_path = f"{glob.MULTICAM_ROOT}/{id_}/annotations/test.json",
            num_img=5,
            no_img_id_offset=True
        )

    validate_jpgs(multicam=True)

    debug_point = ""