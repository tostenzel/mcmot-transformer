"""
Check whether WILDTRACK data in MOT format is correct.

For this purpose, I build on the `generate_coco_from_mot.py` script and the
`check_coco_from_mot` function that visualizes bounding boxes from COCO
annotations on the images to which they are pointing. Hence, we convert the

To reuse these scripts, we need to converted the MOT data back to COCO again.

The data is stored in `data/WILDTRACK/debug_mot`.
"""
import argparse
import configparser
import csv
import json
import os
import shutil

import wildtrack_globals as glob
from wildtrack_shared import check_coco_from_wildtrack

#VIS_THRESHOLD = 0.25


def generate_coco_from_wildtrack_mot(split_name='train', seq_names=None,
                           mot_dir='mot-eval', mots=False, mots_vis=False,
                           frame_range=None, data_root='data/WILDTRACK'):
    """
    Generates COCO data from WILDTRACK in MOT format.
    """
    if frame_range is None:
        frame_range = {'start': 0.0, 'end': 1.0}

    write_dir = "debug_mot"
    mot_path = os.path.join(data_root, mot_dir)
    coco_dir = os.path.join(data_root, write_dir, split_name)

    if os.path.isdir(coco_dir):
        shutil.rmtree(coco_dir)

    if not os.path.exists(coco_dir):
        # Create a new directory because it does not exist
        os.makedirs(coco_dir)

    annotations = {}
    annotations['type'] = 'instances'
    annotations['images'] = []
    annotations['categories'] = [{"supercategory": "person",
                                  "name": "person",
                                  "id": 1}]
    annotations['annotations'] = []

    annotations_dir = os.path.join(os.path.join(data_root, write_dir, 'annotations'))
    if not os.path.isdir(annotations_dir):
        os.mkdir(annotations_dir)
    annotation_file = os.path.join(annotations_dir, f'{split_name}.json')

    # IMAGE FILES
    img_id = 0

    #seqs = sorted(os.listdir(root_split_path))

    #if seqs_names is not None:
    #    seqs = [s for s in seqs if s in seqs_names]
    annotations['sequences'] = seq_names
    annotations['frame_range'] = frame_range
    print(split_name, seq_names)

    for seq in seq_names:
        # CONFIG FILE
        config = configparser.ConfigParser()
        config_file = os.path.join(mot_path, seq, 'seqinfo.ini')

        if os.path.isfile(config_file):
            config.read(config_file)
            img_width = int(config['Sequence']['imWidth'])
            img_height = int(config['Sequence']['imHeight'])
            seq_length = int(config['Sequence']['seqLength'])

        seg_list_dir = os.listdir(os.path.join(mot_path, seq, 'img1'))
        start_frame = int(frame_range['start'] * seq_length)
        end_frame = int(frame_range['end'] * seq_length)
        seg_list_dir = seg_list_dir[start_frame: end_frame]

        print(f"{seq}: {len(seg_list_dir)}/{seq_length}")
        seq_length = len(seg_list_dir)

        for i, img in enumerate(sorted(seg_list_dir)):

            if i == 0:
                first_frame_image_id = img_id

            annotations['images'].append({"file_name": f"{img}",
                                          "height": img_height,
                                          "width": img_width,
                                          "id": img_id,
                                          "frame_id": i,
                                          "seq_length": seq_length,
                                          "first_frame_image_id": first_frame_image_id,
                                          "license": 1,
                                          })

            img_id += 1

            os.symlink(os.path.join(os.getcwd(), mot_path, seq, 'img1', img),
                       os.path.join(coco_dir, img))

    # GT
    annotation_id = 0
    img_file_name_to_id = {
        img_dict['file_name']: img_dict['id']
        for img_dict in annotations['images']}
    for seq in seq_names:
        # GT FILE
        gt_file_path = os.path.join(mot_path, seq, 'gt', f'{seq}_gt.txt')
        if not os.path.isfile(gt_file_path):
            continue

        seq_annotations = []

        seq_annotations_per_frame = {}
        with open(gt_file_path, "r") as gt_file:
            reader = csv.reader(gt_file, delimiter=' ' if mots else ',')

            for row in reader:
                if int(row[6]) == 1 and int(row[7]) == 1:
                    bbox = [float(row[2]), float(row[3]), float(row[4]), float(row[5])]
                    bbox = [int(c) for c in bbox]

                    area = bbox[2] * bbox[3]
                    visibility = float(row[8])
                    frame_id = int(row[0])
                    if seq.endswith("test"):
                        num_prev_imgs = glob.TRAIN_SEQ_LENGTH
                        # fill with zeros on the left
                        wildtrack_img_counter = str(num_prev_imgs*5 + frame_id*5).zfill(8)
                        image_id = img_file_name_to_id.get(
                            f"{seq[0:3]}{wildtrack_img_counter}.jpg",
                            None
                        )
                    elif seq.endswith("val"):
                        num_prev_imgs = glob.TRAIN_SEQ_LENGTH + glob.TEST_SEQ_LENGTH
                        wildtrack_img_counter = str(num_prev_imgs*5 + frame_id*5).zfill(8)
                        image_id = img_file_name_to_id.get(
                            f"{seq[0:3]}{wildtrack_img_counter}.jpg",
                            None
                        )
                    else:
                        raise ValueError
                    if image_id is None:
                        continue
                    track_id = int(row[1])

                    annotation = {
                        "id": annotation_id,
                        "bbox": bbox,
                        "image_id": image_id,
                        "segmentation": [],
                        #"ignore": 0 if visibility > VIS_THRESHOLD else 1,
                        "visibility": visibility,
                        "area": area,
                        "iscrowd": 0,
                        "seq": seq,
                        "category_id": annotations['categories'][0]['id'],
                        "track_id": track_id,
                    }

                    seq_annotations.append(annotation)
                    if frame_id not in seq_annotations_per_frame:
                        seq_annotations_per_frame[frame_id] = []
                    seq_annotations_per_frame[frame_id].append(annotation)

                    annotation_id += 1

            annotations['annotations'].extend(seq_annotations)

    # max objs per image
    num_objs_per_image = {}
    for anno in annotations['annotations']:
        image_id = anno["image_id"]

        if image_id in num_objs_per_image:
            num_objs_per_image[image_id] += 1
        else:
            num_objs_per_image[image_id] = 1

    print(f'max objs per image: {max(list(num_objs_per_image.values()))}')

    with open(annotation_file, 'w') as anno_file:
        json.dump(annotations, anno_file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate COCO from MOT.')
    parser.add_argument('--mots20', action='store_true')
    parser.add_argument('--mot20', action='store_true')
    args = parser.parse_args()

    # we only have to check test and val because train is not used by eval
    # scripts in mot format
    
    seq_names = [s + "-test" for s in glob.SEQUENCE_IDS]
    generate_coco_from_wildtrack_mot(    
        split_name='test',
        seq_names=seq_names)

    seq_names = [s + "-val" for s in glob.SEQUENCE_IDS]
    generate_coco_from_wildtrack_mot(
        split_name='val',
        seq_names=seq_names)  

    check_coco_from_wildtrack(
        img_dir_path = f"{glob.ROOT}/debug_mot/test",
        coco_annotations_path = f"{glob.ROOT}/debug_mot/annotations/test.json",
        split="test",
        write_path = f"{glob.ROOT}/debug_mot/debug_coco_images",
        read_symlinked_symlinked_jpgs = True
    )
    
    check_coco_from_wildtrack(
        img_dir_path = f"{glob.ROOT}/debug_mot/val",
        coco_annotations_path = f"{glob.ROOT}/debug_mot/annotations/val.json",
        split="val",
        write_path = f"{glob.ROOT}/debug_mot/debug_coco_images",
        read_symlinked_symlinked_jpgs = True
    )
