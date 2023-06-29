"""Test `multicam_wildtrack_torch_3D_to_2D.py`"""

import json

import numpy as np
import torch
from cv2 import Rodrigues as cv2Rodrigues
from cv2 import projectPoints as cv2projectPoints

from wildtrack_globals import SRC_ANNS, ANNOTATION_FILES

from wildtrack_shared import convert_wildtrack_to_coco_bbox

from multicam_wildtrack_torch_3D_to_2D import projectPoints
from multicam_wildtrack_torch_3D_to_2D import Rodrigues
from multicam_wildtrack_torch_3D_to_2D import load_spec_extrinsics
from multicam_wildtrack_torch_3D_to_2D import load_spec_intrinsics
from multicam_wildtrack_torch_3D_to_2D import transform_3D_cylinder_to_2D_COCO_bbox_params

from multicam_wildtrack_3D_cylinder_to_2D_bbox_projections import transform_3D_cylinder_to_2D_bbox_params as transform_3D_cylinder_to_2D_bbox_params_np
from multicam_wildtrack_2D_bbox_to_3D_cylinder_projections import decode_3D_cylinder_center
from multicam_wildtrack_2D_bbox_to_3D_cylinder_projections import transform_2D_bbox_to_3D_cylinder_params

from multicam_wildtrack_load_calibration import load_all_intrinsics
from multicam_wildtrack_load_calibration import load_all_extrinsics


#-------------------------------------------------------------------------------
# load bbox data

# load bbox data to project to cylinders
# only test on images from first time period
with open(SRC_ANNS + "/" + ANNOTATION_FILES[0], mode="r", encoding="utf-8") as f:
    data = json.load(f)

#-------------------------------------------------------------------------------
# load_calibration

# numpy
rvecs_np, tvecs_np = load_all_extrinsics()
camera_matrices_np, dist_coeffs_np = load_all_intrinsics()
rvec0_np, tvec0_np, camera_matrix0_np, dist_coeffs0_np = rvecs_np[0], tvecs_np[0], camera_matrices_np[0], dist_coeffs_np[0]

# torch
rvec0, tvec0 = load_spec_extrinsics(view=0)
camera_matrix0, dist_coeffs0 = load_spec_intrinsics(view=0)
camera_matrix1, dist_coeffs1 = load_spec_intrinsics(view=1)

torch.equal(torch.from_numpy(rvec0_np), rvec0)
torch.equal(torch.from_numpy(tvec0_np), tvec0)
torch.equal(torch.from_numpy(camera_matrix0_np), camera_matrix0)
torch.equal(torch.from_numpy(dist_coeffs0_np), dist_coeffs0)

#-------------------------------------------------------------------------------
# Test vs. cv2.Rodrigues

rot_mat = Rodrigues(rvec0)
rot_mat_np = cv2Rodrigues(rvec0_np)[0]

torch.equal(torch.from_numpy(rot_mat_np), rot_mat)

#-------------------------------------------------------------------------------
# Test vs. cv2.projectPoints

# get cylinder from positionID
positionID = data[0]["positionID"]  # data[0]["positionID"]
true_X_foot_np = decode_3D_cylinder_center(positionID)
true_X_foot = torch.from_numpy(true_X_foot_np).reshape((1,3))
stacked_X_foot = torch.cat([true_X_foot, true_X_foot], dim=0)

x = projectPoints(true_X_foot, rvec0, tvec0, camera_matrix0)
stacked_x = projectPoints(stacked_X_foot, rvec0, tvec0, camera_matrix0)
x_np, _ = cv2projectPoints(true_X_foot_np, rvec0_np, tvec0_np, camera_matrix0_np, None)

torch.isclose(x, torch.from_numpy(x_np))

#-------------------------------------------------------------------------------
# test transform_3D_cylinder_to_2D_COCO_bbox_params

# get two cylinders from same camera
cylinder0_dict = transform_2D_bbox_to_3D_cylinder_params(data[0]["views"][0], rvec0_np, tvec0_np, camera_matrix0_np)
cylinder1_dict = transform_2D_bbox_to_3D_cylinder_params(data[1]["views"][0], rvec0_np, tvec0_np, camera_matrix0_np)
cylinder0 = torch.tensor(list(cylinder0_dict.values()), dtype=torch.float32).reshape((1, 4))
cylinder1 = torch.tensor(list(cylinder1_dict.values()), dtype=torch.float32).reshape((1, 4))
cyl_stacked =  torch.cat([cylinder0, cylinder1], dim=0)

bbox0_np = transform_3D_cylinder_to_2D_bbox_params_np(cylinder0_dict, rvec0_np, tvec0_np, camera_matrix0_np, None)
bbox1_np = transform_3D_cylinder_to_2D_bbox_params_np(cylinder1_dict, rvec0_np, tvec0_np, camera_matrix0_np, None)
bbox0_np = convert_wildtrack_to_coco_bbox(**bbox0_np)
bbox1_np = convert_wildtrack_to_coco_bbox(**bbox1_np)
# why type conversion?
bbox_stacked_np = np.vstack((bbox0_np, bbox1_np)).astype(np.float32)
bbox_stacked = transform_3D_cylinder_to_2D_COCO_bbox_params(cyl_stacked, rvec0, tvec0, camera_matrix0)

torch.equal(torch.from_numpy(bbox_stacked_np), bbox_stacked)

#-------------------------------------------------------------------------------
