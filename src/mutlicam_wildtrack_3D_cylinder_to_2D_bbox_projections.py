"""Transform 3D world grid cylinder parameters to 2D image plane bbox params.

Notation follows Duy's thesis, Appendix B in
https://arxiv.org/pdf/2111.11892.pdf.

"""
# pylint: disable=[E1136]

import math
from typing import Dict, Optional

import numpy as np
from cv2 import Rodrigues
from cv2 import projectPoints as project_3D_to_2D

from wildtrack_globals import W, H


def transform_3D_cylinder_to_2D_bbox_params(
    cylinder,
    rvec,
    tvec,
    camera_matrix,
    dist_coeffs
) -> Optional[Dict]:
    """Transforms 3D cylinder to 2D bbox given camera calibration.

    Args:
        cylinder: cylinder data with x_center, y_center, height and radius.
        rvec: camera rotation vector (extrinsic parameters).
        tvec: camera translation vector (extrinsic paramters).
        camera_matrix: camera matrix (intrinsic paramters).
        dist_coeffs: distortion coefficients.

    Returns:
        bbox params.

    """
    R, _ = Rodrigues(rvec)
    Ctilde = (- R.T @ tvec).flatten()
    X_foot = np.float32([cylinder["x_center"], cylinder["y_center"], 0])

    X0_lowerright_corner_temp = _shift_2D_point_perpendicular(
        P1=X_foot[0:2],  # pylint: disable=[E1136]
        P2=Ctilde[0:2],
        dist=cylinder["radius"]
    )

    X0_lowerright_corner = np.zeros(3)
    X0_lowerright_corner[0:2] = X0_lowerright_corner_temp

    x0_lowerright_corner, _ = project_3D_to_2D(
        X0_lowerright_corner,
        rvec.flatten(),
        tvec.flatten(),
        camera_matrix,
        dist_coeffs
    )

    # MINUS radius
    X0_lowerright_corner_temp = _shift_2D_point_perpendicular(
        P1=X_foot[0:2],  # pylint: disable=[E1136]
        P2=Ctilde[0:2],
        dist=-cylinder["radius"]
    )
    X0_upperright_corner = np.zeros(3)
    X0_upperright_corner[0:2] = X0_lowerright_corner_temp
    X0_upperright_corner[2] = cylinder["height"]

    x0_upperright_corner, _ = project_3D_to_2D(
        X0_upperright_corner,
        rvec.flatten(),
        tvec.flatten(),
        camera_matrix,
        dist_coeffs
    )

    bbox =  {
        "xmax": np.round(x0_lowerright_corner.flatten()[0]),
        "xmin": np.round(x0_upperright_corner.flatten()[0]),
        "ymax": np.round(x0_lowerright_corner.flatten()[1]),
        "ymin": np.round(x0_upperright_corner.flatten()[1])
    }

    # Check that whether can see the whole bbox in this camera
    if (W < bbox["xmax"] < 0) or (W < bbox["xmin"] < 0) \
    or (H < bbox["ymax"] < 0) or (H < bbox["ymin"] < 0):
        return None
    else:
        return bbox


def _shift_2D_point_perpendicular(
    P1: np.array,
    P2: np.array,
    dist: float
) -> np.array:
    """Shift 2D point P1 perpendicular to line P1 to P2 in 2D by a distance.
    
    Args:
        P1: 2D Array defining the 2D point that is shifted.
        P2: 2D Array defining the other point.
        dist: distance by which P1 is shifted.

    Returns:
        new_P1: 2D array of the shifted P1.

    """
    x1, y1 = P1
    x2, y2 = P2

    # Calculate the direction vector
    direction_vector = np.float32([x2 - x1, y2 - y1])

    # Calculate the magnitude of the direction vector
    magnitude = math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)

    # Normalize the direction vector
    unit_direction_vector = np.float32([
        direction_vector[0] / magnitude,
        direction_vector[1] / magnitude
    ])

    # Calculate the displacement vector
    displacement_vector = np.float32([
        dist * (-unit_direction_vector[1]),
        dist * unit_direction_vector[0]
    ])

    # Calculate the new position of P1
    new_P1 = np.float32([
        x1 + displacement_vector[0],
        y1 + displacement_vector[1]
    ])

    return new_P1
