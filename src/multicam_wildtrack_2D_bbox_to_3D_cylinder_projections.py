"""Transform 2D image plane bbox parameters to 3D world grid cylinder params.

Notation follows Duy's thesis, Appendix B in
https://arxiv.org/pdf/2111.11892.pdf.

"""
from typing import Dict, Tuple

import numpy as np
from cv2 import Rodrigues
from numpy.linalg import inv#, pinv
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar


def transform_2D_bbox_to_3D_cylinder_params(
    bbox: Dict,
    rvec: np.array,
    tvec: np.array,
    camera_matrix: np.array
) -> Dict:
    """Transforms 2D bbox to 3D cylinder given camera calibration.

    Args:
        bbox: bbox data with xmax, xmin, ymax, ymin,
            origin in upper left corner.
        rvec: camera rotation vector (extrinsic parameters).
        tvec: camera translation vector (extrinsic paramters).
        camera_matrix: camera matrix (intrinsic paramters).

    Returns:
        cylinder params.

    """
    x_foot = (bbox["xmax"] + bbox["xmin"]) / 2
    y_foot = bbox["ymax"]
    X_foot, _ = _project_2D_to_3D(x_foot, y_foot, rvec, tvec, camera_matrix)
    X_center = X_foot[0]
    Y_center = X_foot[1]
    Height = _get_cylinderheight_from_bbox(
        x_foot,
        bbox["ymin"],
        bbox["ymax"],
        rvec,
        tvec,
        camera_matrix
    )
    X_lowerright_corner, _  = _project_2D_to_3D(
        bbox["xmin"],
        bbox["ymax"],
        rvec, tvec,
        camera_matrix
    )
    Radius = np.linalg.norm(X_lowerright_corner[0:2] - X_foot[0:2])
    return {
        "x_center": X_center,
        "y_center": Y_center,
        "height": Height,
        "radius": Radius
    }


def _project_2D_to_3D(
        x: float,
        y: float,
        rvec: np.array,
        tvec: np.array,
        K: np.array
) -> Tuple[np.array, np.array]:
    """Project 2D image plane point to 3D world grid.

    Commented out can test an equivalence relation.

    Args:
        x: image x
        y: image y, origin is top left corner
        rvec: camera rotation vector (extrinsic parameters)
        tvec: camera translation vector (extrinsic paramters)
        K: camera matrix (intrinsic paramters)

    Returns:
        3D projection of image
        camera center in 3D worldcoordinates

    """
    #def get_Rt(R, tvec):
    #    return np.concatenate((R, tvec), axis=1)

    x_homogenous = np.float32([x, y, 1]).reshape((3,1)) # pylint: disable=[E1121]
    R, _ = Rodrigues(rvec)
    #Rt = get_Rt(R, tvec)
    #P = K @ Rt
    M = K @ R

    # Ctilde coordinates of the camera centre C in the world coordinate
    Ctilde = - R.T @ tvec
    #np.testing.assert_almost_equal(
    #    Ctilde,
    #    - inv(M) @ P[:,3].reshape((3,1)), decimal=4
    #)
    #Pplus0 = pinv(P)
    Xtilde = inv(M) @ x_homogenous
    Xtilde_pi = - (Ctilde[2] / Xtilde[2]) * (inv(M) @ x_homogenous) + Ctilde

    return Xtilde_pi.flatten(), Ctilde.flatten()


def _decode_3D_cylinder_center(id_: int) -> np.array:
    """Decode WILDTRACK positionID into 3d world grid foot point.

    Taken from Wildtrack Readme but with intercept and coefficient times
    scale := 100.
    2.5cm is one grid point step, 480 is image height.

    Args:
        id_: WILDTRACK id that encodes x and y in (x,y,0) in 3D world grid.

    Returns:
        3D world coordinates of bbox object with respective id.

    """
    x_grid = -300 + 2.5 * (id_ % 480)
    y_grid = -900 + 2.5 * (id_ / 480)
    return np.float32([[x_grid, y_grid, 0]])


def _get_cylinderheight_from_bbox(x_avg, ymin, ymax, rvec, tvec, K):
    """Compute heigth of 3D cylinder from bbox parameters.

    Args:
        x_avg: bbox midpoint horizontal
        ymin: bbox minimum vertical (left side on img)
        ymax: bbox maxvertical (right side on img)
        rvec: camera rotation vector (extrinsic parameters)
        tvec: camera translation vector (extrinsic paramters)
        K: camera matrix (intrinsic paramters)

    Returns:
        height of 3D cylinder projected from 2D bbox params

    """
    # get 2D image upper bbox center on 3D floor and
    # camera center in 3D coordinates
    X_head_floor, Ctilde  = _project_2D_to_3D(x_avg, ymin, rvec, tvec, K)
    X_foot, _  = _project_2D_to_3D(x_avg, ymax, rvec, tvec, K)


    # Minimize the objective function starting from an initial guess
    #z4_initial_guess = 180 / 2.5 # 1.80m in 3d grid units
    z4_minimized = get_height_that_minimizes_distance(Ctilde, X_foot, X_head_floor)
    return z4_minimized


def get_height_that_minimizes_distance(
    P1: np.array,
    P2: np.array,
    P3: np.array,
    #z4_initial_guess: float
) -> float:
    """Compute distance between 3D point and line between two 3D points.

    More specifically, the 3D point P4=(x2, y2, z4) is above P2=(x2,y2,z2=0) on
    the ground plane and the line is between P1=(x1,y1,z1) and P3=(x3,y3,z3=0),
    also on the ground plane.

    """
    def distance_to_line(z4):
        # Calculate the direction vector of the line
        direction_vector = np.float32([P3[0] - P1[0], P3[1] - P1[1], P3[2] - P1[2]])

        # Calculate the magnitude of the direction vector
        magnitude = np.linalg.norm(direction_vector)

        # Normalize the direction vector
        unit_direction_vector = direction_vector / magnitude

        # Calculate the point P4
        P4 = np.float32([P2[0], P2[1], z4])

        # Calculate the vector from P1 to P4
        vector_P1_P4 = P4 - P1

        # Calculate the distance between the line and the point P2
        distance = np.linalg.norm(np.cross(unit_direction_vector, vector_P1_P4))

        return distance
    
    fn_evals = []
    from scipy.stats import norm

    dist = norm(loc=173, scale=7)
    bounds = dist.cdf([173 - 7*2.5, 173 + 7*2.5])
    pp = np.linspace(*bounds, num=10)
    args = dist.ppf(pp)
    
    for i in args:

    # Evaluate the scalar function at each argument
        fn_evals.append(distance_to_line(i/2.5))

        # Find the index of the minimum value in the function_values array
        min_index = np.argmin(np.array([fn_evals]))

    # Get the argument corresponding to the minimum value
    min_argument = args[min_index]
    #return min_argument
    return min_argument