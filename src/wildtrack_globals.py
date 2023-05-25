"""Global variables for WILDTRACK modules to be imported from here."""

ROOT = "data/WILDTRACK"
MULTICAM_ROOT = "data/multicam_WILDTRACK"

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