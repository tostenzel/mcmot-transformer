"""Generate MOT data for tracking eval from COCO data.

Follows `wildtrack_generate_mot_from_coco.py`.

"""

from wildtrack_globals import MULTICAM_ROOT# = "data/multicam_WILDTRACK"


from wildtrack_generate_mot_from_coco import generate_mot_from_coco


COCO_ROOT = MULTICAM_ROOT + "/c0"


if __name__ == "__main__":
    generate_mot_from_coco(coco_root=COCO_ROOT, multicam=True)