"""Global variables for WILDTRACK modules to be imported from here."""

import os

# ------------------------------------------------------------------------------
# WILDTRACK format

SEQ_LENGTH = 400
W, H = 1920, 1080
ANNOTATED_FPS = 2

# one wildtrack 3D world grid unit corresponds to 2.5 cm
CM_TO_3D_WORLD = 2.5

#-------------------------------------------------------------------------------
# PATH settings

# original Wildtrack
SRC_ANNS = "data/Wildtrack_dataset/annotations_positions"
SRC_IMG = os.path.join(os.path.dirname(SRC_ANNS), "Image_subsets")
SRC_CALIBRATION = "data/Wildtrack_dataset/calibrations"
# where we store our preprocessed data
ROOT = "data/WILDTRACK"
MULTICAM_ROOT = "data/multicam_WILDTRACK"

# original WILDTRACK annotation variables
ANNOTATION_FILES = [
    file for file in os.listdir(SRC_ANNS) if file.endswith(".json")
    ]
ANNOTATION_FILES.sort()
N_ANNOTATIONS = len(ANNOTATION_FILES)

#-------------------------------------------------------------------------------
# my settings
TRAIN_SPLIT = 5 / 400
TEST_SPLIT =  5 / 400
SEQUENCE_IDS = ["c0", "c1"]#, "c2", "c3", "c4", "c5", "c6"]
N_CAMS = len(SEQUENCE_IDS)

# dependent variables
TRAIN_SEQ_LENGTH = round(SEQ_LENGTH * TRAIN_SPLIT)
TEST_SEQ_LENGTH = round(SEQ_LENGTH * TEST_SPLIT)
VAL_SEQ_LENGTH = round(SEQ_LENGTH * (1 - TRAIN_SPLIT - TEST_SPLIT))

#-------------------------------------------------------------------------------
# calibration files

EXTRINSIC_CALIBRATION_FILES = [
    "extr_CVLab1.xml",
    "extr_CVLab2.xml",
    "extr_CVLab3.xml",
    "extr_CVLab4.xml",
    "extr_IDIAP1.xml",
    "extr_IDIAP2.xml",
    "extr_IDIAP3.xml"
]

INTRINSIC_CALIBRATION_FILES = [
    "intr_CVLab1.xml",
    "intr_CVLab2.xml",
    "intr_CVLab3.xml",
    "intr_CVLab4.xml",
    "intr_IDIAP1.xml",
    "intr_IDIAP2.xml",
    "intr_IDIAP3.xml"
]
#-------------------------------------------------------------------------------

# train cylinder stats with not enough values (REDO)

X_CENTER = {"min": -305.0 ,"max": 907.0}
Y_CENTER = {"min": -862.0 ,"max": 2479.0}
HEIGHT = {"min": 173.0 ,"max": 178.6}
RADIUS = {"min": 13.0,"max": 52.0}
#-------------------------------------------------------------------------------

