"""
Generate `data/multicam_WILDTRACK´ (.jpg) from orig. `data/Wildtrack_dataset` (.png).

Annotations are only in the first sequence folder

"""
from typing import List

import copy
import json
import os
import tqdm

import numpy as np
from PIL import Image

import wildtrack_globals as glob
from wildtrack_shared import check_coco_from_wildtrack, COCO_BASE_DICT
from wildtrack_shared import validate_jpgs
from multicam_wildtrack_load_calibration import load_all_intrinsics
from multicam_wildtrack_load_calibration import load_all_extrinsics

# Now destination paths inside cam folder
for id_ in glob.SEQUENCE_IDS:
    DEST_COCO_ANNOTATIONS = f"{glob.MULTICAM_ROOT}/{id_}/annotations"
    if os.path.isdir(DEST_COCO_ANNOTATIONS) is False:
        os.makedirs(DEST_COCO_ANNOTATIONS)

# load list of rvec and tvecs and camera matrices
rvecs, tvecs = load_all_extrinsics()
camera_matrices, dist_coeffs = load_all_intrinsics()

def generate_3D_coco_from_wildtrack() -> None:
    """
    Create one single-camera tracking coco WILDTRACKdataset with seven sequences.
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
    train_ann_id, test_ann_id, val_ann_id = 0, 0, 0
    output_train_annotation = "train.json"
    output_test_annotation = "test.json"
    output_val_annotation = "val.json"
    for c in range(glob.N_CAMS):

        train_images, train_annotations, train_ann_id = create_annotations(
            train_annotation_files, c, "train", train_ann_id
            )
        test_images, test_annotations, test_ann_id = create_annotations(
            test_annotation_files, c, "test", test_ann_id
            )
        val_images, val_annotations, val_ann_id = create_annotations(
            val_annotation_files, c, "val", val_ann_id
            )

        DEST_COCO_TRAIN = f"{glob.MULTICAM_ROOT}/{glob.SEQUENCE_IDS[c]}/train"
        DEST_COCO_TEST = f"{glob.MULTICAM_ROOT}/{glob.SEQUENCE_IDS[c]}/test"
        DEST_COCO_VAL = f"{glob.MULTICAM_ROOT}/{glob.SEQUENCE_IDS[c]}/val"

        for d in [DEST_COCO_TRAIN, DEST_COCO_TEST, DEST_COCO_VAL]:
            if os.path.isdir(d) is False:
                os.mkdir(d)

        create_coco_files(
            c,
            train_dataset,
            train_images,
            train_annotations,
            output_train_annotation,
            DEST_COCO_TRAIN
            )
        create_coco_files(
            c,
            test_dataset,
            test_images,
            test_annotations,
            output_test_annotation,
            DEST_COCO_TEST
            )
        create_coco_files(
            c,
            val_dataset,
            val_images,
            val_annotations,
            output_val_annotation,
            DEST_COCO_VAL
            )


def create_annotations(
        ann_files: List[dict],
        c: int,
        split: str="train",
        start_annotation_id: int = 0
        ) -> tuple([List[dict], List[dict]]):
    """Creates annotations for every object on each image of a single-camera train, test or validation split.

    This function is used in function `main` in a loop over the number of cameras.
    WILDTRACK uses the same image and annotations ids for each camera.
    We have to seperate the ids with offset variables.
    Originally, each sequence has length 400. Yet, we use each of the seven
    sequences for training, test and validation data. In each split, we count the
    image id from 0 to 7 times the split length.

    annotation_id has to be a unique id for every bbox etc in the folder.
    Therefore, it has to be different for all camera subsets. To implement this
    as part of a for loop over the cameras that calls this function,
    we have to start with the last annotation ID from the last camera and count
    up and return

    Args:
        ann_files: WILDTRACK annotation files for this split. 
            However, one file contains annotations for every view.
        c: index variable for camera id starting from 0.
        split: flag to indicate whether we should use `TEST_SEQ_LENGHT` or
            `VAL_SEQ_LENGHT` instead.
        start_annotation_id: unique annotation id for the whole dataset.

    Returns:
        images: list of immage infos, esp. tracking specific info like
            frame_id, seq_length, and first frame of seq.
        annotations: list of annotations for one object, esp. bbox, ann and
            img ids, and tracking specific info such like track_id and seq id.
        ann_id: Last annotation id to start from in the next function call.
    """
    # It seems that all IDs have to start at 0 due to trackformer indexing
    if split == "train":
        seq_name_appendix = "-train"
        # Perhaps bug in old dataset
        seq_length = glob.TRAIN_SEQ_LENGTH
    elif split =="test":
        seq_name_appendix = "-test"
        seq_length = glob.TEST_SEQ_LENGTH
    elif split == "val":
        seq_name_appendix = "-val"
        seq_length = glob.VAL_SEQ_LENGTH

    #annotation_id_offset = c * N_ANNOTATIONS
    img_id = 0
    ann_id = start_annotation_id#0#c * N_ANNOTATIONS
    images = []
    annotations = []

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

        from multicam_wildtrack_2D_bbox_to_3D_cylinder_projections import transform_2D_bbox_to_3D_cylinder_params as get_cylinder
        if c==0:
            for instance in data:
                cylinder_list = []
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
                        #print(cylinder)
                        cylinder_arr = np.fromiter(cylinder.values(), dtype=float)
                        cylinder_list.append(cylinder_arr)
                    
                cylinder_mean = np.mean(cylinder_list, axis=0)
                #print(cylinder_mean)
                # FIXME: Problem: I get same data for all timeperiods
                cylinder_list = []
                annotations.append({
                    "id": ann_id,# + annotation_id_offset,
                    "bbox": [
                        # TODO TOBIAS: Perhaps not round
                        cylinder_mean[0],
                        cylinder_mean[1],
                        cylinder_mean[2],
                        cylinder_mean[3]
                        ],
                    "image_id": img_id,# + img_id_offset, #+ val_img_id_offset,
                    "segmentation": [],
                    #"ignore":,
                    "visibility": 1.0,
                    # TOBIAS: changes from w_box * h_box
                    "area": None,
                    "category_id": 1,
                    "iscrowd": 0,
                    # tracking specific
                    "seq": f"c{c}" + seq_name_appendix,
                    # TODO: perhaps offset, too? Yet, this info should make baseline stronger.
                    "track_id": instance["personID"]
                })

                ann_id += 1
        img_id += 1

    return images, annotations, ann_id


def create_coco_files(
        c: int,
        dataset: dict,
        images: List[dict],
        annotations: List[dict],
        dest_coco_dict: str,
        dest_img_files: str
    ) -> None:
    """
    Stores annotations as .json, and converts and stores images for one train or val split.
    Also writes image and object annotations into whole dataset annotation.
    Args:
        dataset: COCO_BASE_DICT.
        images: image annotations
        annotations: object annotations
        dest_coco_dict: folder for complete annotation .json file 
        dest_img_files: folder for image files
    """
    dataset["images"] = images
    dataset["annotations"] = annotations

    json.dump(dataset, open(f"{glob.MULTICAM_ROOT}/{glob.SEQUENCE_IDS[c]}/annotations/{dest_coco_dict}", "w"), indent=4)

    for img in tqdm.tqdm(dataset["images"]):
        src_file_name = img["file_name"].rsplit("-", 1)[1].rsplit(".", 1)[0] + ".png"
        cam = img["file_name"].rsplit("-", 1)[0] # e.g. "c0" for accessing the "C1" folder
        full_file_name = os.path.join(glob.SRC_IMG, f"C{int(cam[1])+1}", src_file_name)
        im = Image.open(full_file_name)
        rgb_im = im.convert("RGB")

        # save .jpg
        pic_path = os.path.join(
            dest_img_files, img["file_name"]
            )
        rgb_im.save(pic_path)
        im.save(pic_path)


if __name__ == "__main__":
    generate_3D_coco_from_wildtrack()

    # annotation path must be fixed to c0 and converted to sequence in check
    # to load the right image?

    for id_ in glob.SEQUENCE_IDS:
        check_coco_from_wildtrack(
            img_dir_path = f"{glob.MULTICAM_ROOT}/{id_}/train",
            write_path = f"{glob.MULTICAM_ROOT}/debug_coco_images",
            # Fix to c0
            coco_annotations_path = f"{glob.MULTICAM_ROOT}/c0/annotations/train.json",
            multicam=True,
            three_dim=True,
            num_img=5
        )
        
        #check_coco_from_wildtrack(
        #    img_dir_path = f"{glob.MULTICAM_ROOT}/{id_}/test",
        #    write_path = f"{glob.MULTICAM_ROOT}/debug_coco_images",
        #    coco_annotations_path = f"{glob.MULTICAM_ROOT}/{id_}/annotations/test.json",
        #    multicam=True,
        #    three_dim=True,
        #    num_img=5
        #)
        #check_coco_from_wildtrack(
        #    img_dir_path = f"{glob.MULTICAM_ROOT}/{id_}/val",
        #    write_path = f"{glob.MULTICAM_ROOT}/debug_coco_images",
        #    coco_annotations_path = f"{glob.MULTICAM_ROOT}/{id_}/annotations/val.json",
        #    multicam=True,
        #    three_dim=True,
        #    num_img=5
        #)
        

    validate_jpgs(multicam=True)

    debug_point = ""