"""
Generate `data/WILDTRACK´ (.jpg) from orig. `data/Wildtrack_dataset` (.png).
This dataset is a single-camera dataset with seven constructed in contrast to
the original seven-identical-camera WILDTRACK dataset.
Important related files are the original WILDTRACK dataset and
`generate_coco_from_mot.py`.
bbox in COCO format: (x,y,h,w), 
 - x: number of pixels from the left image border to the (upper) left pixel
 - y: number of pixel from the top image border the to the upper (left) pixel
 - h: height of bbox
 - w: width of bbox
Thus, (x,y) are the coordinates of the top left corner of the bbox viewed from
an origin at the top left corner of the image.
Apparently, WILDTRACK has (xmin, ymin, xmax, ymax) with origin at (0,0). This
means (x,y) = (W-xmax, H-ymax).This
[line](https://github.com/Chavdarova/WILDTRACK-toolkit/blob/master/annotations_viewer.py#L314) indicates the point.
We do not need 1-indexing like Eval modules and MOT format.
`generat_coco_from_mot.py` does also use 0-indexing.
"""
from typing import List

import copy
import json
import os
import tqdm
from PIL import Image

import wildtrack_globals as glob
# debug dependencies
from pycocotools.coco import COCO
import skimage.io as io
from matplotlib import pyplot as plt


# WILDTRACK format (7 cameras, one frame size, one sequence length)
# we have to insert the camera views as sequences one by one and pretend that
# they are unrelated.

# Source paths
SRC_ANNS = "data/Wildtrack_dataset/annotations_positions"
SRC_IMG = os.path.join(os.path.dirname(SRC_ANNS), "Image_subsets")

# get number of annotation files (contains data on all cams) to later construct
# an offset for distributing these annotations equally for one single cam each
ANNOTATION_FILES = [
    file for file in os.listdir(SRC_ANNS) if file.endswith(".json")
    ]
ANNOTATION_FILES.sort()
N_ANNOTATIONS = len(ANNOTATION_FILES)

# destination paths
DEST_COCO_ANNOTATIONS = f"{glob.ROOT}/annotations"
if os.path.isdir(DEST_COCO_ANNOTATIONS) is False:
    os.makedirs(DEST_COCO_ANNOTATIONS)

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
    "frame_range": {'start': 0.0, 'end': 1.0},
    "sequences": None
}

def flatten_listoflists(ll: List[list]) -> list:
    l = []
    for sublist in ll:
        l.extend(sublist)
    return l


def main() -> None:
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

    number_train_files = int(glob.TRAIN_SPLIT*len(ANNOTATION_FILES))
    number_test_files = int(glob.TEST_SPLIT*len(ANNOTATION_FILES))

    train_annotation_files = ANNOTATION_FILES[:number_train_files]
    test_annotation_files = ANNOTATION_FILES[number_train_files:(number_train_files + number_test_files)]
    val_annotation_files = ANNOTATION_FILES[(number_train_files + number_test_files):]

    mc_data = {
        "train_images": [],
        "train_annotations": [],
        "test_images": [],
        "test_annotations": [],
        "val_images": [],
        "val_annotations": []
        }

    # flexible annotation id for uneven annotation number per camera and sequence
    train_ann_id, test_ann_id, val_ann_id = 0, 0, 0
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

        mc_data["train_images"].append(train_images)
        mc_data["train_annotations"].append(train_annotations)
        mc_data["test_images"].append(test_images)
        mc_data["test_annotations"].append(test_annotations)
        mc_data["val_images"].append(val_images)
        mc_data["val_annotations"].append(val_annotations)
   
    for key in mc_data:
        mc_data[key] = flatten_listoflists(mc_data[key])

    output_train_annotation = f"train.json"
    output_test_annotation = f"test.json" 
    output_val_annotation = f"val.json"

    DEST_COCO_TRAIN = f"{glob.ROOT}/train"
    DEST_COCO_TEST = f"{glob.ROOT}/test"
    DEST_COCO_VAL = f"{glob.ROOT}/val"
    
    for d in [DEST_COCO_TRAIN, DEST_COCO_TEST, DEST_COCO_VAL]:
        if os.path.isdir(d) is False:
            os.mkdir(d)

    create_coco_files(
        train_dataset,
        mc_data["train_images"],
        mc_data["train_annotations"],
        output_train_annotation,
        DEST_COCO_TRAIN
        )
    create_coco_files(
        test_dataset,
        mc_data["test_images"],
        mc_data["test_annotations"],
        output_test_annotation,
        DEST_COCO_TEST
        )
    create_coco_files(
        val_dataset,
        mc_data["val_images"],
        mc_data["val_annotations"],
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
        img_id_offset = c * glob.TRAIN_SEQ_LENGTH
        seq_name_appendix = "-train"
    elif split =="test":
        img_id_offset = c * glob.TEST_SEQ_LENGTH
        seq_name_appendix = "-test"
    elif split == "val":
        img_id_offset = c * glob.VAL_SEQ_LENGTH
        seq_name_appendix = "-val"

    #annotation_id_offset = c * N_ANNOTATIONS
    img_id = 0
    ann_id = start_annotation_id#0#c * N_ANNOTATIONS
    images = []
    annotations = []

    for ann_file in ann_files:
        data = json.load(open(SRC_ANNS + "/" + ann_file, 'r'))  # was .json

        image_name = ann_file.rsplit('.', 1)[0] + ".jpg"
        image_name = f"c{c}-" + ann_file.rsplit('.', 1)[0] + ".jpg"
        images.append({
            "file_name": f"{image_name}",
            "height": glob.H,
            "width": glob.W,
            "id": img_id + img_id_offset,
            "license": 1,
            # tracking specific
            # `frame_id` is the img's position relative to its sequence,
            # not the whole dataset (0 - 400),
            # see https://github.com/timmeinhardt/trackformer/issues/33#issuecomment-1105108004
            # Starts from 1 in MOT format
            "frame_id": img_id,
            "seq_length": glob.TRAIN_SEQ_LENGTH,
            "first_frame_image_id": 0 + img_id_offset
        })

        for instance in data:
            # only use data for the selected camera and not for all others,
            # where the same person is also visible
            xmax, ymax, xmin, ymin = instance["views"][c]['xmax'], instance["views"][c]['ymax'], instance["views"][c][
            'xmin'], instance["views"][c]['ymin']
            if not (xmax == -1 or ymax == -1 or xmin == -1 or ymin == 1):
                x = xmin
                y = ymin
                w_box = xmax - xmin
                h_box = ymax - ymin
                annotations.append({
                    "id": ann_id,# + annotation_id_offset,
                    "bbox": [
                        int(round(x)),
                        int(round(y)),
                        int(round(w_box)),
                        int(round(h_box))
                        ],
                    "image_id": img_id + img_id_offset, #+ val_img_id_offset,
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
   
    return images, annotations, ann_id


def create_coco_files(
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
    dataset['images'] = images
    dataset['annotations'] = annotations

    json.dump(dataset, open(DEST_COCO_ANNOTATIONS + "/" + dest_coco_dict, 'w'), indent=4)

    for img in tqdm.tqdm(dataset['images']):
        src_file_name = img["file_name"].rsplit('-', 1)[1].rsplit('.', 1)[0] + ".png"
        cam = img["file_name"].rsplit('-', 1)[0] # e.g. "c0" for accessing the "C1" folder
        full_file_name = os.path.join(SRC_IMG, f"C{int(cam[1])+1}", src_file_name)
        im = Image.open(full_file_name)
        rgb_im = im.convert('RGB')

        # save .jpg
        pic_path = os.path.join(
            dest_img_files, img["file_name"]
            )
        rgb_im.save(pic_path)
        im.save(pic_path)


def check_coco_from_wildtrack(
        img_dir_path: str = f"{glob.ROOT}/train",
        coco_annotations_path: str = f"{DEST_COCO_ANNOTATIONS}/train.json",
        split: str = "train"
    ) -> None:
    """
    Visualizes and stores generated COCO data. Only used for debugging.
    We save 3 files for each camera in data/WILDTRACK/debug_images.
    `validation_data` flag has to be true if we pass the validation data
    directories.
    """
    if os.path.isdir("data/WILDTRACK/debug_images") is False:
        os.makedirs("data/WILDTRACK/debug_images")
    
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

        for img_id in range(0, 20):

            img_id = img_id + c * img_id_offset
            img_annotation = coco.loadImgs(img_id)[0]
            i = io.imread(img_dir_path + "/" + img_annotation['file_name'])

            plt.imshow(i)
            plt.axis('off')
            ann_ids = coco.getAnnIds(
                imgIds=img_annotation['id'],
                catIds=cat_ids,
                iscrowd=None
                )
            anns = coco.loadAnns(ann_ids)
            coco.showAnns(anns, draw_bbox=True)
            plt.savefig(f'data/WILDTRACK/debug_images/debug_{img_annotation["file_name"]}')
            # clear figures/bboxes for next picture
            plt.clf()
            debug_point = ""


if __name__ == '__main__':
    main()
    check_coco_from_wildtrack()
    check_coco_from_wildtrack(
        img_dir_path = f"{glob.ROOT}/test",
        coco_annotations_path = f"{DEST_COCO_ANNOTATIONS}/test.json",
        split="test"
    )
    check_coco_from_wildtrack(
        img_dir_path = f"{glob.ROOT}/val",
        coco_annotations_path = f"{DEST_COCO_ANNOTATIONS}/val.json",
        split="val"
    )

    debug_point = ""