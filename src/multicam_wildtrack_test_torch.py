"""Test `multicam_wildtrack_torch_3D_to_2D.py`"""

import json

import torch

from cv2 import Rodrigues as cv2Rodrigues
from cv2 import projectPoints

from multicam_wildtrack_torch_3D_to_2D import projectPoint
from multicam_wildtrack_torch_3D_to_2D import Rodrigues
from multicam_wildtrack_torch_3D_to_2D import load_spec_extrinsics
from multicam_wildtrack_torch_3D_to_2D import load_spec_intrinsics
from multicam_wildtrack_torch_3D_to_2D import transform_3D_cylinder_to_2D_bbox_params
from multicam_wildtrack_3D_cylinder_to_2D_bbox_projections import transform_3D_cylinder_to_2D_bbox_params as transform_3D_cylinder_to_2D_bbox_params_np
from multicam_wildtrack_2D_bbox_to_3D_cylinder_projections import decode_3D_cylinder_center
from multicam_wildtrack_2D_bbox_to_3D_cylinder_projections import transform_2D_bbox_to_3D_cylinder_params

from multicam_wildtrack_load_calibration import load_all_intrinsics
from multicam_wildtrack_load_calibration import load_all_extrinsics

from wildtrack_globals import SRC_ANNS, ANNOTATION_FILES


#-------------------------------------------------------------------------------
# load data

# load bbox data to project to cylinders
# only test on images from first time period
with open(SRC_ANNS + "/" + ANNOTATION_FILES[0], mode="r", encoding="utf-8") as f:
    data = json.load(f)


# load_calibration

# numpy
rvecs_np, tvecs_np = load_all_extrinsics()
camera_matrices_np, dist_coeffs_np = load_all_intrinsics()
rvec0_np, tvec0_np, camera_matrix0_np, dist_coeffs0_np = rvecs_np[0], tvecs_np[0], camera_matrices_np[0], dist_coeffs_np[0]
rvec1_np, tvec1_np, camera_matrix1_np, dist_coeffs1_np = rvecs_np[1], tvecs_np[1], camera_matrices_np[1], dist_coeffs_np[1]


# torch
rvec0, tvec0 = load_spec_extrinsics(view=0)
camera_matrix0, dist_coeffs0 = load_spec_intrinsics(view=0)
rvec1, tvec1 = load_spec_extrinsics(view=1)
camera_matrix1, dist_coeffs1 = load_spec_intrinsics(view=1)

torch.equal(torch.from_numpy(rvec0_np), rvec0)
torch.equal(torch.from_numpy(tvec0_np), tvec0)
torch.equal(torch.from_numpy(camera_matrix0_np), camera_matrix0)
torch.equal(torch.from_numpy(dist_coeffs0_np), dist_coeffs0)

#-------------------------------------------------------------------------------
# cv2


rot_mat = Rodrigues(rvec0)
rot_mat_np = cv2Rodrigues(rvec0_np)[0]
torch.equal(torch.from_numpy(rot_mat_np), rot_mat)

# only test on images from first time period
with open(SRC_ANNS + "/" + ANNOTATION_FILES[0], mode="r", encoding="utf-8") as f:
    data = json.load(f)

positionID = data[0]["positionID"]  # data[0]["positionID"]
true_X_foot_np = decode_3D_cylinder_center(positionID)
true_X_foot = torch.from_numpy(true_X_foot_np)
x = projectPoint(true_X_foot, rvec0, tvec0, camera_matrix0)
x_np, _ = projectPoints(true_X_foot_np, rvec0_np, tvec0_np, camera_matrix0_np, None)
torch.isclose(x, torch.from_numpy(x_np))
#-------------------------------------------------------------------------------

cylinder0_dict = transform_2D_bbox_to_3D_cylinder_params(data[0]["views"][0], rvec0_np, tvec0_np, camera_matrix0_np)
cylinder1_dict = transform_2D_bbox_to_3D_cylinder_params(data[0]["views"][1], rvec1_np, tvec1_np, camera_matrix1_np)

cylinder0 = torch.tensor(list(cylinder0_dict.values()), dtype=torch.float32)
cylinder1 = torch.tensor(list(cylinder1_dict.values()), dtype=torch.float32)

bbox0_np = transform_3D_cylinder_to_2D_bbox_params_np(cylinder0_dict, rvec0_np, tvec0_np, camera_matrix0_np, None)
bbox1_np = transform_3D_cylinder_to_2D_bbox_params_np(cylinder1_dict, rvec1_np, tvec1_np, camera_matrix1_np, None)

bbox0 = transform_3D_cylinder_to_2D_bbox_params(cylinder0, rvec0, tvec0, camera_matrix0)
bbox1 = transform_3D_cylinder_to_2D_bbox_params(cylinder1, rvec1, tvec1, camera_matrix1)

print("np: ", bbox0_np, bbox1_np)

print("torch: ", bbox0, bbox1)
debug = "db"
