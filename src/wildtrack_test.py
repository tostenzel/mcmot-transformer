"""Test equality between the image and tracking annotations in COCO and MOT.

The data is WILDTRACK in COCO format and in the COCO format that was converted
from the above COCO data to MOT and then back to COCO again. This is done to
reuse the function `wildtrack_shared.check_coco_from_wildtrack` that visualizes
bounding boxes on images from data in COCO format.

There are some ordering and annotation id differences that shoud not matter.

"""
import json

import pandas as pd

import wildtrack_globals as glob


EVAL_SPLITS = ["test", "val"]

def test_annotations():
    for split in EVAL_SPLITS:
        coco_anns = f"{glob.ROOT}/annotations/{split}.json"
        # annotations from coco to mot to coco conversions
        coco_from_mot_anns = f"{glob.ROOT}/debug_mot/annotations/{split}.json"

        with open(coco_anns) as json_file:
            coco_dict = json.load(json_file)
        with open(coco_from_mot_anns) as json_file:
            debug_dict = json.load(json_file)

        assert coco_dict["images"] == debug_dict["images"]

        # "id" column and column order different
        coco_df = pd.DataFrame.from_dict(coco_dict["annotations"])
        coco_df.drop("id", axis=1)
        coco_df.sort_values(
            ['seq', 'image_id', 'track_id'],
            ascending=[True, True, True]
        ).reset_index(drop=True)
        
        debug_df = pd.DataFrame.from_dict(debug_dict["annotations"])
        debug_df.drop("id", axis=1)
        debug_df.sort_values(
            ['seq', 'image_id', 'track_id'],
            ascending=[True, True, True]
        ).reset_index(drop=True)
        
        pd.testing.assert_frame_equal(coco_df,debug_df.reindex(columns=coco_df.columns))
