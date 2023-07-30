"""Efficient reimplementation of `multicam_wildtrack_2D_bbox_rto_3D_cylinder_projections.py`.

!!! Necessary for giving matched bboxes from track queries as references point
deformable transformer!!!

Vectorized, completely in PyTorch with bbox return value already in COCO format
instead of WILDTRACK format.

Will be used for the evaluation in `engine.py`.

Notation follows Duy's thesis, Appendix B in
https://arxiv.org/pdf/2111.11892.pdf.

Large cap X's denote 3D world coordinates, small cap x's 2D camera points.

"""
from typing import Dict, Tuple

import torch

from multicam_wildtrack_torch_3D_to_2D import Rodrigues


def transform_2D_bbox_to_3D_cylinder_params_batch(
    bbox: torch.Tensor,
    rvec: torch.Tensor,
    tvec: torch.Tensor,
    camera_matrix: torch.Tensor,
    device: str
) -> Dict:
    """Transforms 2D bbox to 3D cylinder given camera calibration.

    Args:
        bbox: Bbox data in xmin, ymin, xmax, ymax format
        Shape: (N, 4)
        ...

    Returns:
        Cylinder data.
        Shape: (N, 4)

    """
    x_foot = (bbox[:, 2] + bbox[:, 0]) / 2#(bbox["xmax"] + bbox["xmin"]) / 2
    y_foot = bbox[:, 3] #bbox["ymax"]
    X_foot, _ = _project_2D_to_3D(x_foot, y_foot, rvec, tvec, camera_matrix)
    X_center = X_foot[:, 0]
    Y_center = X_foot[:, 1]
    Height = torch.full((bbox.shape[0],), 178.6)
    X_lowerright_corner, _ = _project_2D_to_3D(
        bbox[:, 0],#bbox["xmin"],
        bbox[:, 3],#bbox["ymax"],
        rvec, tvec,
        camera_matrix
    )
    Radius = torch.norm(X_lowerright_corner[:, 0:2] - X_foot[:, 0:2], dim=1)
    cyl_batch = torch.stack([X_center, Y_center, Height, Radius], dim=1)

    return cyl_batch


def _project_2D_to_3D(
        x: torch.Tensor,
        y: torch.Tensor,
        rvec: torch.Tensor,
        tvec: torch.Tensor,
        K: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project batch of 2D image plane points to 3D world grid.

    Args:
        x: image x. Shape: (N,)
        y: image y, origin is top left corner. Shape: (N,)
        rvec: camera rotation vector (extrinsic parameters). Shape: (3,)
        tvec: camera translation vector (extrinsic parameters). Shape: (3,)
        K: camera matrix (intrinsic parameters). Shape: (3, 3)

    Returns:
        3D projection of images. Shape: (N, 3)
        camera center in 3D world coordinates. Shape: (3,)

    """
    dtype = torch.float32

    # Convert x and y to homogeneous coordinates
    x_homogeneous = torch.stack([x, y, torch.ones_like(x, dtype=dtype)], dim=1)

    # Convert rvec to a single rotation matrix R
    R = Rodrigues(rvec)

    # Compute camera matrices M (batch-wise)
    M = torch.matmul(K, R)

    # Ctilde coordinates of the camera centre C in the world coordinate
    Ctilde = -torch.matmul(R.T, tvec)

    # Compute Xtilde (batch-wise)
    Xtilde = torch.matmul(torch.inverse(M), x_homogeneous.unsqueeze(-1)).squeeze(-1)

    # Compute Xtilde_pi (batch-wise)
    Xtilde_pi = -(Ctilde[2] / Xtilde[:, 2]).unsqueeze(-1) * Xtilde + Ctilde.squeeze(1)

    return Xtilde_pi[:, 0:2], Ctilde


