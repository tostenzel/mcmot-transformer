"""
Generate `data/multicam_WILDTRACK´ (.jpg) from orig. `data/Wildtrack_dataset` (.png).

"""
from typing import List

import copy
import json
import os
import tqdm
from PIL import Image

import wildtrack_globals as glob


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

# Move inside cam folder

# destination paths
for id in glob.SEQUENCE_IDS:
    DEST_COCO_ANNOTATIONS = f"{glob.MULTICAM_ROOT}/{id}/annotations"
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


    # flexible annotation id for uneven annotation number per camera and sequence
    train_ann_id, test_ann_id, val_ann_id = 0, 0, 0
    output_train_annotation = f"train.json"
    output_test_annotation = f"test.json" 
    output_val_annotation = f"val.json"
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
        data = json.load(open(SRC_ANNS + "/" + ann_file, 'r'))  # was .json

        image_name = ann_file.rsplit('.', 1)[0] + ".jpg"
        image_name = f"c{c}-" + ann_file.rsplit('.', 1)[0] + ".jpg"
        images.append({
            "file_name": f"{image_name}",
            "height": glob.H,
            "width": glob.W,
            "id": img_id,# + img_id_offset,
            "license": 1,
            # tracking specific
            # `frame_id` is the img's position relative to its sequence,
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
    dataset['images'] = images
    dataset['annotations'] = annotations

    json.dump(dataset, open(f"{glob.MULTICAM_ROOT}/{glob.SEQUENCE_IDS[c]}/annotations/{dest_coco_dict}", 'w'), indent=4)

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


def validate_jpgs():
    """
    Validate converted .jpgs.
    
    Sometimes some files were not valid and caused errors during trainig.
    Code according to
    https://stackoverflow.com/questions/46854496/python-script-to-detect-broken-images
    """
    for id in glob.SEQUENCE_IDS:
        for split in ["train", "test", "val"]:
            path = f'{glob.MULTICAM_ROOT}/{id}/{split}'
            for filename in os.listdir(path):
                if filename.endswith('.jpg'):
                    try:
                        im = Image.open(f"{path}/{filename}")
                        im.verify() #I perform also verify, don't know if he sees other types o defects
                        im.close() #reload is necessary in my case
                        im = Image.open(f"{path}/{filename}")
                        im.transpose(Image.FLIP_LEFT_RIGHT)
                        im.close()
                    except (IOError, SyntaxError) as e:
                        print(filename)


from pycocotools.coco import COCO
import skimage.io as io
from matplotlib import pyplot as plt

def check_multicam_coco_from_wildtrack(
        img_dir_path: str,
        coco_annotations_path: str,
        write_path = f"{glob.MULTICAM_ROOT}/debug_coco_images",
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
       
    for img_id in range(0, num_img):

        img_id = img_id
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


if __name__ == '__main__':
    #main()
    for id in glob.SEQUENCE_IDS:
        check_multicam_coco_from_wildtrack(
            img_dir_path = f"{glob.MULTICAM_ROOT}/{id}/train",
            coco_annotations_path = f"{glob.MULTICAM_ROOT}/{id}/annotations/train.json",
        )
        check_multicam_coco_from_wildtrack(
            img_dir_path = f"{glob.MULTICAM_ROOT}/{id}/test",
            coco_annotations_path = f"{glob.MULTICAM_ROOT}/{id}/annotations/test.json",
        )
        check_multicam_coco_from_wildtrack(
            img_dir_path = f"{glob.MULTICAM_ROOT}/{id}/val",
            coco_annotations_path = f"{glob.MULTICAM_ROOT}/{id}/annotations/val.json",
        )

    validate_jpgs()

    debug_point = ""