"""Load WILDTRACK configurations for working in 3D."""

import numpy as np
from xml.dom import minidom
from os.path import isfile
import xml.etree.ElementTree as ElementTree

from wildtrack_globals import INTRINSIC_CALIBRATION_FILES, EXTRINSIC_CALIBRATION_FILES, CALIBRATION_ROOT


def load_opencv_xml(filename: str, element_name: str, dtype: str='float32') -> np.array:
    """
    Loads particular element from a given OpenCV XML file.

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
        return np.fromstring(tree.find(element_name).find('data').text,
                             dtype, count=rows*cols, sep=' ').reshape((rows, cols))
    except Exception as e:
        print(e)
        raise UnicodeDecodeError('Error while decoding file %s.' % filename)


def load_all_extrinsics() -> tuple:
    """
    Loads all the extrinsic files.

    Returns:
        tuple of ([2D array], [2D array]) where the first and the second integers
            are indexing the camera/file and the element of the corresponding vector,
            respectively. E.g. rvec[i][j], refers to the rvec for the i-th camera,
            and the j-th element of it (out of total 3)
    """
    rvec, tvec = [], []
    for _file in EXTRINSIC_CALIBRATION_FILES:
        xmldoc = minidom.parse(CALIBRATION_ROOT + '/extrinsic/' + _file)
        rvec.append([float(number)
                     for number in xmldoc.getElementsByTagName('rvec')[0].childNodes[0].nodeValue.strip().split()])
        tvec.append([float(number)
                     for number in xmldoc.getElementsByTagName('tvec')[0].childNodes[0].nodeValue.strip().split()])
        
    for i in range(0, len(rvec)):
        rvec[i] = np.array(rvec[i], dtype=np.float32).reshape((3, 1))
        tvec[i] = np.array(tvec[i], dtype=np.float32).reshape((3, 1))

    return rvec, tvec


def load_all_intrinsics():
    """
    Loads all the intrinsic files.

    :param _lst_files: [str] path of a file listing all the intrinsic calibration files
    :return: tuple (cameraMatrices[list], distortionCoef[list])
    """
    _cameraMatrices, _distCoeffs = [], []
    for _file in INTRINSIC_CALIBRATION_FILES:
        _cameraMatrices.append(load_opencv_xml(CALIBRATION_ROOT + '/intrinsic_zero/' + _file, 'camera_matrix'))
        _distCoeffs.append(load_opencv_xml(CALIBRATION_ROOT + '/intrinsic_zero/' + _file, 'distortion_coefficients'))
    return _cameraMatrices, _distCoeffs
