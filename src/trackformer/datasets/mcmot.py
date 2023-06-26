"""Build multicam dataset.

It has the same training-relevant function signatures like the single-cam
datasets (list of MOT instances) but outputs data for multiple
cameras (see __get_item__).

"""
from typing import List

from .mot import MOT


class MCMOT():
    def __init__(self, singlecam_MOT_datasets: List[MOT]):
        self.datasets = singlecam_MOT_datasets
    #--------------------------------------------------------------------------
    # overwrite class functions from MOT class
    @property
    def sequences(self):
        return ["complete WILDTRACK"]
    # ['c0-train', 'c1-train']

    @property
    def frame_range(self):
        return self.datasets[0].frame_range
    # {'start': 0.0, 'end': 1.0}

    def seq_length(self, idx):
        return self.datasets[0].seq_length(idx)
    # returns 40 for idx in 0 to 39

    def sample_weight(self, idx):
        return self.datasets[0].sample_weight(idx)

    def __getitem__(self, idx):
        # return list instead of image tensor and coco dict
        img = [ds.__getitem__(idx)[0] for ds in self.datasets]
        target = [ds.__getitem__(idx)[1] for ds in self.datasets]

        # Why do image tensors have different sizes? Some transformations beforehand?
        return img, target

    #---------------------------------------------------------------------------
    # overwrite class functions from CoCoDetection
    def __len__(self) -> int:
        return self.datasets[0].__len__()
    #---------------------------------------------------------------------------
