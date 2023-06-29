"""Reimplements `multicam_wildtrack_3D_cylinder_to_2D_bbox_projections.py`
completely in PyTorch, with bbox return value in COCO format instead WILDTRACK.

Will be used for the evaluation in `engine.py`.

Notation follows Duy's thesis, Appendix B in
https://arxiv.org/pdf/2111.11892.pdf.

Large cap X's denote 3D world coordinates, small cap x's 2D camera points.

"""
from typing import Tuple

from xml.dom import minidom
import torch

from multicam_wildtrack_load_calibration import _load_opencv_xml
from wildtrack_globals import INTRINSIC_CALIBRATION_FILES, \
    EXTRINSIC_CALIBRATION_FILES, SRC_CALIBRATION
from wildtrack_globals import W, H



def transform_3D_cylinder_to_2D_bbox_params(
    cylinder: torch.Tensor,
    rvec: torch.Tensor,
    tvec: torch.Tensor,
    camera_matrix: torch.Tensor,
) -> torch.Tensor:
    """Transforms a 3D cylinder to 2D bbox given camera calibration.

    Args:
        cylinder: Cylinder data with x_center, y_center, height, and radius.
        Shape: (4,)
        ...

    Returns:
        Bbox parameters.
        Shape: (4,)

    """

    R = Rodrigues(rvec)
    Ctilde = (-torch.matmul(R.T, tvec)).flatten()
    X_foot = torch.tensor([cylinder[0], cylinder[1], 0], dtype=torch.float32)

    X0_lowerright_corner_temp = _shift_2D_point_perpendicular(
        P1=X_foot[0:2],
        P2=Ctilde[0:2],
        dist=cylinder[3]
    )

    X0_lowerright_corner = torch.zeros(3)
    X0_lowerright_corner[0:2] = X0_lowerright_corner_temp

    x0_lowerright_corner = projectPoint(
        X=X0_lowerright_corner,
        rvec=rvec,
        tvec=tvec,
        camera_matrix=camera_matrix
    )
    print("torch: x0_lowerright_corner", x0_lowerright_corner)

    # MINUS radius
    X0_lowerright_corner_temp = _shift_2D_point_perpendicular(
        P1=X_foot[0:2],
        P2=Ctilde[0:2],
        dist=-cylinder[3]
    )
    X0_upperright_corner = torch.zeros(3)
    X0_upperright_corner[0:2] = X0_lowerright_corner_temp
    X0_upperright_corner[2] = cylinder[2]

    x0_upperright_corner = projectPoint(
        X=X0_upperright_corner,
        rvec=rvec,
        tvec=tvec,
        camera_matrix=camera_matrix
    )

    bbox = torch.tensor([
        torch.round(x0_lowerright_corner.flatten()[0]),
        torch.round(x0_upperright_corner.flatten()[0]),
        torch.round(x0_lowerright_corner.flatten()[1]),
        torch.round(x0_upperright_corner.flatten()[1])
    ], dtype=torch.float32)

    # Check whether we can see the whole bbox in this camera
    if (W < bbox[0] < 0) or (W < bbox[1] < 0) \
    or (H < bbox[2] < 0) or (H < bbox[3] < 0):
        return torch.tensor([-1, -1, -1, -1], dtype=torch.float32)
    else:
        return bbox



def _shift_2D_point_perpendicular(
        P1: torch.Tensor,
        P2: torch.Tensor,
        dist: torch.Tensor
) -> torch.Tensor:
    """
    Shift a 2D point P1 perpendicular to the line from P1 to P2 in 2D by a given distance.

    """

    x1, y1 = P1
    x2, y2 = P2

    # Calculate the direction vector
    direction_vector = torch.tensor([x2 - x1, y2 - y1], dtype=torch.float32)

    # Calculate the magnitude of the direction vector
    magnitude = torch.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)

    # Normalize the direction vector
    unit_direction_vector = direction_vector / magnitude

    # Calculate the displacement vector
    displacement_vector = torch.tensor([
        dist * (-unit_direction_vector[1]),
        dist * unit_direction_vector[0]
    ], dtype=torch.float32)

    # Calculate the new position of P1
    new_P1 = torch.tensor([
        x1 + displacement_vector[0],
        y1 + displacement_vector[1]
    ], dtype=torch.float32)

    return new_P1


def projectPoint(
    X: torch.Tensor,
    rvec: torch.Tensor,
    tvec: torch.Tensor,
    camera_matrix: torch.Tensor
) -> torch.Tensor:
    """Project a 3D point onto the image plane similar to cv2.projectPoints.

    Returns:
        Projected 2D point on the image plane.
        Shape: (2,)
    """
    # Create a homogeneous representation of the 3D point
    X_homo = torch.ones((4, 1), dtype=torch.float32)
    X_homo[0:3, 0] = X

    # Convert the rotation vector to a rotation matrix
    R = Rodrigues(rvec)

    # Concatenate the rotation matrix and translation vector
    Rt_inhomo = torch.cat((R, tvec), dim=1)

    # Compute the projection matrix
    P = camera_matrix @ Rt_inhomo

    # Project the point onto the image plane
    x_homo = P @ X_homo
    x = x_homo[:2] / x_homo[2]  # Normalize the homogeneous coordinates

    return x.flatten()


def Rodrigues(rvec: torch.Tensor) -> torch.Tensor:
    """Get rotation matrix from vector similar to cv2.Rodrigues.

    Args:
        rvec: Rotation vector of shape (3,).

    Returns:
        Rotation matrix of shape (3, 3).
    """
    theta = torch.norm(rvec)
    if theta < 1e-8:
        return torch.eye(3, dtype=torch.float32)

    r = rvec / theta
    c = torch.cos(theta)
    s = torch.sin(theta)
    v = 1 - c

    rmat = torch.tensor([[r[0] * r[0] * v + c,     r[0] * r[1] * v - r[2] * s, r[0] * r[2] * v + r[1] * s],
                         [r[1] * r[0] * v + r[2] * s, r[1] * r[1] * v + c,     r[1] * r[2] * v - r[0] * s],
                         [r[2] * r[0] * v - r[1] * s, r[2] * r[1] * v + r[0] * s, r[2] * r[2] * v + c]],
                        dtype=torch.float32)

    return rmat


def load_spec_extrinsics(view: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Loads extrinsic files for one specific view."""
    _file = EXTRINSIC_CALIBRATION_FILES[view]
    xmldoc = minidom.parse(SRC_CALIBRATION + '/extrinsic/' + _file)
    rvec = [float(number)
                for number in xmldoc.getElementsByTagName('rvec')[0].childNodes[0].nodeValue.strip().split()]
    tvec = [float(number)
                for number in xmldoc.getElementsByTagName('tvec')[0].childNodes[0].nodeValue.strip().split()]
    
    rvec = torch.tensor(rvec, dtype=torch.float32).reshape((3, 1))
    tvec = torch.tensor(tvec, dtype=torch.float32).reshape((3, 1))

    return rvec, tvec


def load_spec_intrinsics(view: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Loads intrinsic files for one specific view."""

    _file = INTRINSIC_CALIBRATION_FILES[view]
    camera_matrix = _load_opencv_xml(
        SRC_CALIBRATION + '/intrinsic_zero/' + _file, 'camera_matrix'
    )
    distortion_coefficient = _load_opencv_xml(
        SRC_CALIBRATION + '/intrinsic_zero/' + _file, 'distortion_coefficients'
    )
    return torch.from_numpy(camera_matrix), torch.from_numpy(distortion_coefficient)





