
import math

import numpy as np
from cv2 import Rodrigues
from cv2 import projectPoints as project_3D_to_2D



def transform_3D_cylinder_to_2D_bbox_params(cylinder, rvec, tvec, camera_matrix, distcoeffs):
    
    R, _ = Rodrigues(rvec)
    Ctilde = (- R.T @ tvec).flatten()
    X_foot = np.float32([cylinder["x_center"], cylinder["y_center"], 0])

    X0_lowerright_corner_temp = shift_2D_point_perpendicular(P1=X_foot[0:2], P2=Ctilde[0:2], dist=cylinder["radius"])
    X0_lowerright_corner = np.zeros(3)
    X0_lowerright_corner[0:2] = X0_lowerright_corner_temp

    x0_lowerright_corner, _ = project_3D_to_2D(
        X0_lowerright_corner,  # 3D points
        rvec.flatten(),  # rotation rvec
        tvec.flatten(),  # translation tvec
        camera_matrix,
        distcoeffs  # camera matrix
    )

    # MINUS radius
    X0_lowerright_corner_temp = shift_2D_point_perpendicular(P1=X_foot[0:2], P2=Ctilde[0:2], dist=-cylinder["radius"])
    X0_upperright_corner = np.zeros(3)
    X0_upperright_corner[0:2] = X0_lowerright_corner_temp
    X0_upperright_corner[2] = cylinder["height"]

    x0_upperright_corner, _ = project_3D_to_2D(
        X0_upperright_corner ,  # 3D points
        rvec.flatten(),  # rotation rvec
        tvec.flatten(),  # translation tvec
        camera_matrix,
        distcoeffs  # camera matrix
    )

    return {
        "xmax": np.round(x0_lowerright_corner.flatten()[0]),
        "xmin": np.round(x0_upperright_corner.flatten()[0]),
        "ymax": np.round(x0_lowerright_corner.flatten()[1]),
        "ymin": np.round(x0_upperright_corner.flatten()[1])
    }


def shift_2D_point_perpendicular(P1: np.array, P2: np.array, dist: float) -> np.array:
    """Shift 2D point P1 perpendicular to line from P1 to P2 in 2D by some distance.
    
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
    direction_vector = (x2 - x1, y2 - y1)

    # Calculate the magnitude of the direction vector
    magnitude = math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)

    # Normalize the direction vector
    unit_direction_vector = (direction_vector[0] / magnitude, direction_vector[1] / magnitude)

    # Calculate the displacement vector
    displacement_vector = (dist * (-unit_direction_vector[1]), dist * unit_direction_vector[0])

    # Calculate the new position of P1
    new_P1 = np.array([x1 + displacement_vector[0], y1 + displacement_vector[1]])

    return new_P1
