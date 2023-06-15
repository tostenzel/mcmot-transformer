"""Global variables for WILDTRACK modules to be imported from here."""
import os

ROOT = "data/WILDTRACK"
MULTICAM_ROOT = "data/multicam_WILDTRACK"
CALIBRATION_ROOT = "data/Wildtrack_dataset/calibrations"
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

# format
SEQ_LENGTH = 400
W, H = 1920, 1080

# my settings
TRAIN_SPLIT = 40 / 400
TEST_SPLIT =  40 / 400
SEQUENCE_IDS = ["c0", "c1"]#, "c2", "c3", "c4", "c5", "c6"]
N_CAMS = len(SEQUENCE_IDS)

# dependent variables
TRAIN_SEQ_LENGTH = round(SEQ_LENGTH * TRAIN_SPLIT)
TEST_SEQ_LENGTH = round(SEQ_LENGTH * TEST_SPLIT)
VAL_SEQ_LENGTH = round(SEQ_LENGTH * (1 - TRAIN_SPLIT - TEST_SPLIT))

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
