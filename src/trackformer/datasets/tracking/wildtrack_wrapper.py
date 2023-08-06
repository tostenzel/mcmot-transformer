"""
WILDTRACK wrapper for one train or val sequence/camera.

"""
from torch.utils.data import Dataset

from .wildtrack_sequence import WILDTRACKSequence


class WILDTRACKWrapper(Dataset):
    """A Wrapper for the WILDTRACK class to return multiple sequences."""
    # no dets kwarg for wildtrack!
    def __init__(self, split: str, cam: str, **kwargs) -> None:
        """Initliazes WILDTRACK sequences.

        Keyword arguments:
        split: train, test or val-
        cam: camera in {0,...6}-
        kwargs -- kwargs for the WILDTRACK sequence.
        """
        self._data = []
        # only append the respective sequence once
        self._data.append(WILDTRACKSequence(seq_name=f"{cam}-{split}", **kwargs))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]