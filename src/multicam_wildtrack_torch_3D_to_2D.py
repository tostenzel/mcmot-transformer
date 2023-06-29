"""Efficient reimplementation of `multicam_wildtrack_3D_cylinder_to_2D_bbox_projections.py`.

Vectorized, completely in PyTorch with bbox return value already in COCO format
instead of WILDTRACK format.

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


def transform_3D_cylinder_to_2D_COCO_bbox_params(
    cylinder: torch.Tensor,
    rvec: torch.Tensor,
    tvec: torch.Tensor,
    camera_matrix: torch.Tensor,
    device: str
) -> torch.Tensor:
    """Transforms N 3D cylinders to 2D bbox in COCO given camera calibration.

    If a cylinder can not be seen in the image, return (-1, -1, -1, -1).

    Args:
        cylinder: Cylinder data with x_center, y_center, height, and radius.
        Shape: (N, 4)
        ...

    Returns:
        Bbox parameters.
        Shape: (N, 4)

    """

    # Move tensors to the GPU device
    cylinder = cylinder.to(device)
    rvec = rvec.to(device)
    tvec = tvec.to(device)
    camera_matrix = camera_matrix.to(device)

    R = Rodrigues(rvec).to(device)
    Ctilde = (-torch.matmul(R.T, tvec)).flatten()
    X_foot = torch.stack([
        cylinder[:, 0],
        cylinder[:, 1],
        torch.zeros(cylinder.shape[0]).to(device)
        ], dim=1).to(device)

    X0_lowerright_corner_temp = _shift_2D_points_perpendicular(
        P1=X_foot[:, :2],
        P2=Ctilde[:2],  # cam center always the same
        dist=cylinder[:, 3]
    )

    X0_lowerright_corner = torch.stack([
        X0_lowerright_corner_temp[:, 0],
        X0_lowerright_corner_temp[:, 1],
        torch.zeros(cylinder.shape[0]).to(device)
        ], dim=1).to(device)

    x0_lowerright_corner = projectPoints(
        X=X0_lowerright_corner,
        rvec=rvec,
        tvec=tvec,
        camera_matrix=camera_matrix,
        device=device
    )

    # MINUS radius
    X0_lowerright_corner_temp = _shift_2D_points_perpendicular(
        P1=X_foot[:, :2],
        P2=Ctilde[:2],
        dist=-cylinder[:, 3]
    )

    X0_upperright_corner = torch.stack([
        X0_lowerright_corner_temp[:, 0],
        X0_lowerright_corner_temp[:, 1],
        cylinder[:, 2]
        ],
        dim=1).to(device)

    x0_upperright_corner = projectPoints(
        X=X0_upperright_corner,
        rvec=rvec,
        tvec=tvec,
        camera_matrix=camera_matrix,
        device=device
    )

    # see convert_wildtrack_to_coco_bbox in wildtrack_shared.py
    # xmin, ymin, width, height_box, origin at upper left corner of image
    bbox = torch.stack([
        x0_upperright_corner[:, 0],
        x0_upperright_corner[:, 1],
        x0_lowerright_corner[:, 0] - x0_upperright_corner[:, 0],
        x0_lowerright_corner[:, 1] - x0_upperright_corner[:, 1]
    ]).T

    #---------------------------------------------------------------------------
    # Conditions if a cylinder is not visible in the image of a camera

    # xmin, ymin must be positive, height and width strictly positive
    condition_1 = (bbox[:, 0] < 0) | (bbox[:, 1] < 0) | (bbox[:, 2] < 1) | (bbox[:, 3] < 1)
    # xmin or xmin + width > W
    condition_2 = bbox[:, 0] + bbox[:, 2] > W
    # ymin or ymin + height > H
    condition_3 = bbox[:, 1] + bbox[:, 3] > H

    # Check if any condition is violated along the row dimension
    violated = torch.any(torch.stack([condition_1, condition_2, condition_3], dim=1), dim=1)

    # Set rows to (-1, -1, -1, -1) where conditions are violated
    bbox = torch.where(violated.unsqueeze(1), torch.tensor([-1., -1., -1., -1.], dtype=torch.float32).to(device), bbox)
    #---------------------------------------------------------------------------

    return torch.round(bbox)


def convert_wildtrack_to_coco_bbox(xmax, xmin, ymax, ymin):
    """Converts the bbox format used in WILDTRACK to COCO.
    
    Both assume that the origin of an image is the upper left pixel.
    The x and y coordinates for coco represent the upper left bbox corner.
    
    """
    x = xmin
    y = ymin
    width = xmax - xmin
    height_box = ymax - ymin
    return x, y, width, height_box


def _shift_2D_points_perpendicular(
    P1: torch.Tensor,
    P2: torch.Tensor,   # Cam center always the same
    dist: torch.Tensor
) -> torch.Tensor:
    """
    Shift 2D points P1 perpendicular to the line from P1 to P2 in 2D by a given distance.

    Args:
        P1: Tensor of shape (N, 2) representing the coordinates of N 2D points.
        P2: Flat tensor of shape (2,) representing the coordinates of the corresponding point.
        dist: Tensor of shape (N,) representing the distance to shift each point.

    Returns:
        Tensor of shape (N, 2) representing the new coordinates of the shifted points.

    """
    x1, y1 = P1.unbind(dim=1)
    x2, y2 = P2[0], P2[1]

    # Calculate the direction vectors
    direction_vectors = torch.stack([x2 - x1, y2 - y1], dim=1)

    # Calculate the magnitudes of the direction vectors
    magnitudes = torch.norm(direction_vectors, dim=1)

    # Normalize the direction vectors
    unit_direction_vectors = direction_vectors / magnitudes.unsqueeze(1)

    # Calculate the displacement vectors
    displacement_vectors = torch.stack([
        dist * (-unit_direction_vectors[:, 1]),
        dist * unit_direction_vectors[:, 0]
    ], dim=1)

    # Calculate the new positions of P1
    new_P1 = P1 + displacement_vectors

    return new_P1


def projectPoints(
    X: torch.Tensor,
    rvec: torch.Tensor,
    tvec: torch.Tensor,
    camera_matrix: torch.Tensor,
    device
) -> torch.Tensor:
    """Project 3D points onto the image plane similar to cv2.projectPoints.

    Args:
        X: 3D points to project. Shape: (N, 3), where N is the number of points.
        rvec: Camera rotation vector (extrinsic parameters). Shape: (3, 1).
        tvec: Camera translation vector (extrinsic parameters). Shape: (3, 1).
        camera_matrix: Camera matrix (intrinsic parameters). Shape: (3, 3).
        device: device (lol)

    Returns:
        Projected 2D points on the image plane. Shape: (N, 2), where N is the number of points.

    """
    # Create homogeneous representations of the 3D points
    X_homo = torch.ones((X.shape[0], 4), dtype=torch.float32).to(device)
    X_homo[:, :3] = X

    # Convert the rotation vector to a rotation matrix
    R = Rodrigues(rvec).to(device)

    # Concatenate the rotation matrix and translation vector
    Rt_inhomo = torch.cat((R, tvec), dim=1)

    # Compute the projection matrix
    P = (camera_matrix @ Rt_inhomo).to(device)

    # Project the points onto the image plane
    x_homo = torch.matmul(X_homo, P.t())
    x = x_homo[:, :2] / x_homo[:, 2].unsqueeze(1)  # Normalize the homogeneous coordinates

    return x


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
