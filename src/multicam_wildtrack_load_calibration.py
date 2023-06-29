"""Load WILDTRACK camra configurations.

The intrinsic and extrincis camera parameters are necessary for translating
2D bounding boxes from multiple views to 3D cylinders and back.

"""
from typing import List, Tuple

from xml.dom import minidom
from os.path import isfile
import xml.etree.ElementTree as ElementTree

import numpy as np

from wildtrack_globals import INTRINSIC_CALIBRATION_FILES, \
    EXTRINSIC_CALIBRATION_FILES, SRC_CALIBRATION


def load_all_extrinsics() -> Tuple[np.array, np.array]:
    """Loads all the extrinsic files.
    
    Returns tuple of ([2D array], [2D array]) where the first and the second
    integers are indexing the camera/file and the element of the
    corresponding vector, respectively. E.g. rvec[i][j], refers to the rvec
    for the i-th camera, and the j-th element of it (out of total 3).
    
    Returns:
        rvecs: rotation vectors from view 0 to 6
        tvecs: translation vectors

    """
    rvecs, tvecs = [], []
    for _file in EXTRINSIC_CALIBRATION_FILES:
        xmldoc = minidom.parse(SRC_CALIBRATION + '/extrinsic/' + _file)
        rvecs.append([float(number)
                     for number in xmldoc.getElementsByTagName('rvec')[0].childNodes[0].nodeValue.strip().split()])
        tvecs.append([float(number)
                     for number in xmldoc.getElementsByTagName('tvec')[0].childNodes[0].nodeValue.strip().split()])
        
    for i in range(0, len(rvecs)):
        rvecs[i] = np.array(rvecs[i], dtype=np.float32).reshape((3, 1))
        tvecs[i] = np.array(tvecs[i], dtype=np.float32).reshape((3, 1))

    return rvecs, tvecs


def load_all_intrinsics() -> Tuple[List[np.array], List[np.array]]:
    """Loads all the intrinsic files.

    Returns:
        camera_matrices
        distortion_coefficients

    """
    camera_matrices, distortion_coefficients = [], []
    for _file in INTRINSIC_CALIBRATION_FILES:
        camera_matrices.append(
            _load_opencv_xml(SRC_CALIBRATION + '/intrinsic_zero/' + _file, 'camera_matrix')
        )
        distortion_coefficients.append(
            _load_opencv_xml(SRC_CALIBRATION + '/intrinsic_zero/' + _file, 'distortion_coefficients')
            )
    return camera_matrices, distortion_coefficients


def _load_opencv_xml(
        filename: str,
        element_name: str,
        dtype: str='float32'
) -> np.array:
    """Loads particular element from a given OpenCV XML file.

    Raises:
        FileNotFoundError: the given file cannot be read/found
        UnicodeDecodeError: if error occurs while decoding the file

    Args:
        filename: name of the OpenCV XML file
        element_name: element in the file
        dtype: type of element, default: 'float32'

    Returns:
        the value of the element_name

    """
    if not isfile(filename):
        raise FileNotFoundError("File %s not found." % filename)
    try:
        tree = ElementTree.parse(filename)
        rows = int(tree.find(element_name).find('rows').text)
        cols = int(tree.find(element_name).find('cols').text)
        return np.fromstring(
            tree.find(element_name).find('data').text,
            dtype,
            count=rows*cols,
            sep=' '
        ).reshape((rows, cols))
    except Exception:
        raise UnicodeDecodeError('Error while decoding file %s.' % filename)
