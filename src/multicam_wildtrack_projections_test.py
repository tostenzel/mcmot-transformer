
"""Test projection functions from 2D (bbox) to 3D (cylinder) and from 3D to 2D.

Notation follows Duy's paper/thesis, Appendix B in
https://arxiv.org/pdf/2111.11892.pdf.

Large cap X's denote 3D world coordinates, small cap x's 2D camera points.

"""
# pylint: disable=[C0301]
import json

import numpy as np
from cv2 import projectPoints as project_3D_to_2D

from wildtrack_globals import SRC_ANNS, ANNOTATION_FILES

from multicam_wildtrack_load_calibration import load_all_intrinsics
from multicam_wildtrack_load_calibration import load_all_extrinsics

from multicam_wildtrack_2D_bbox_to_3D_cylinder_projections import transform_2D_bbox_to_3D_cylinder_params
from multicam_wildtrack_2D_bbox_to_3D_cylinder_projections import _project_2D_to_3D
from multicam_wildtrack_2D_bbox_to_3D_cylinder_projections import decode_3D_cylinder_center
from multicam_wildtrack_2D_bbox_to_3D_cylinder_projections import _get_cylinderheight_from_bbox

from multicam_wildtrack_3D_cylinder_to_2D_bbox_projections import transform_3D_cylinder_to_2D_bbox_params
from multicam_wildtrack_3D_cylinder_to_2D_bbox_projections import _shift_2D_point_perpendicular


# only test on images from first time period
with open(SRC_ANNS + "/" + ANNOTATION_FILES[0], mode="r", encoding="utf-8") as f:
    data = json.load(f)

# load list of rvec and tvecs and camera matrices
rvec, tvec = load_all_extrinsics()
camera_matrices, dist_coeffs = load_all_intrinsics()

################################################################################
# grab test bbox data ##########################################################

# Note: bbox data has coordinate origin in UPPER left corner instead of
# lower left corner

# view 0, person ID 122, period 0
positionID = data[0]["positionID"]  # 456826

xmax0 = data[0]["views"][0]["xmax"] # 1561
ymax0 = data[0]["views"][0]["ymax"] # 299

xmin0 = data[0]["views"][0]["xmin"] # 1510
ymin0 = data[0]["views"][0]["ymin"] # 139

# view 1, rest same

xmax1 = data[0]["views"][1]["xmax"] # 891
ymax1 = data[0]["views"][1]["ymax"] # 289

xmin1 = data[0]["views"][1]["xmin"] # 813
ymin1 = data[0]["views"][1]["ymin"] # 10


################################################################################
# Test `transform_2D_bbox_to_3D_cylindern_params` ##############################

cylinder0 = transform_2D_bbox_to_3D_cylinder_params(data[0]["views"][0], rvec[0], tvec[0], camera_matrices[0])
cylinder1 = transform_2D_bbox_to_3D_cylinder_params(data[0]["views"][1], rvec[1], tvec[1], camera_matrices[1])
cylinder2 = transform_2D_bbox_to_3D_cylinder_params(data[0]["views"][2], rvec[2], tvec[2], camera_matrices[2])

print(cylinder0, "\n", cylinder1,"\n" , cylinder2)


################################################################################
# Test `_project_2D_to_3D` #####################################################

# get feet position from x center
x0_foot = (xmax0 + xmin0) / 2
y0_foot = ymax0
true_X_foot = decode_3D_cylinder_center(positionID)
X0_foot, Ctilde0  = _project_2D_to_3D(x0_foot, y0_foot, rvec[0], tvec[0], camera_matrices[0])

x1_foot = (xmax1 + xmin1) / 2
y1_foot = ymax1
true_X_foot = decode_3D_cylinder_center(positionID)
X1_foot, _  = _project_2D_to_3D(x1_foot, y1_foot, rvec[1], tvec[1], camera_matrices[1])

print("World coordinates should all be equal...(pos, cam 0->3d, cam1->3d):", true_X_foot, X0_foot, X1_foot ,"\n")


################################################################################
# Test `_decode_3D_cylinder_center` ############################################

x0_twotime_projection, _ = project_3D_to_2D(
        X0_foot.flatten(),  # 3D points
        rvec[0].flatten(),  # rotation rvec
        tvec[0].flatten(),  # translation tvec
        camera_matrices
[0],  # camera matrix
        dist_coeffs[0])  # distortion coefficients

x0_onetime_projection, _ = project_3D_to_2D(
        true_X_foot,  # 3D points
        rvec[0].flatten(),  # rotation rvec
        tvec[0].flatten(),  # translation tvec
        camera_matrices
[0],  # camera matrix
        dist_coeffs[0])  # distortion coefficients

x1_twotime_projection, _ = project_3D_to_2D(
        X1_foot.flatten(),  # 3D points
        rvec[1].flatten(),  # rotation rvec
        tvec[1].flatten(),  # translation tvec
        camera_matrices
[1],  # camera matrix
        dist_coeffs[1])  # distortion coefficients

x1_onetime_projection, _ = project_3D_to_2D(
        true_X_foot,  # 3D points
        rvec[1].flatten(),  # rotation rvec
        tvec[1].flatten(),  # translation tvec
        camera_matrices
[1],  # camera matrix
        dist_coeffs[1])  # distortion coefficients

print("Cam 0 coordinates (pos->2d, 2d->3d->2d, true data):", x0_onetime_projection, x0_twotime_projection, np.array([x0_foot, y0_foot]), "\n")
print("Cam 1 coordinates (pos->2d, 2d->3d->2d, true data):", x1_onetime_projection, x1_twotime_projection, np.array([x1_foot, y1_foot]), "\n")


################################################################################
# Test `get_cylinder_height_from_bbox` #########################################

# Get the optimized value of z4
H0 = _get_cylinderheight_from_bbox(x0_foot, ymin0, ymax0, rvec[0], tvec[0], camera_matrices[0])
X0_head = np.copy(X0_foot)
X0_head[2] = H0
x0_head_test, _ = project_3D_to_2D(
        X0_head,  # 3D points
        rvec[0].flatten(),  # rotation rvec
        tvec[0].flatten(),  # translation tvec
        camera_matrices
[0],  # camera matrix
        dist_coeffs[0])  # distortion coefficients

H1 = _get_cylinderheight_from_bbox(x1_foot, ymin1, ymax1, rvec[1], tvec[1], camera_matrices[1])
X1_head = np.copy(X1_foot)
X1_head[2] = H1
x1_head_test, _ = project_3D_to_2D(
        X1_head,  # 3D points
        rvec[1].flatten(),  # rotation rvec
        tvec[1].flatten(),  # translation tvec
        camera_matrices
[1],  # camera matrix
        dist_coeffs[1])  # distortion coefficients

print(H0, H1)
print("bbox height (the objective here) almost not distored, but width a bit more.")
print("Cam 0: bbox upper center (2d->3d->2d, true)", x0_head_test, np.array([x0_foot, ymin0]))
print("Cam 1: bbox upper center (2d->3d->2d, true)", x1_head_test, np.array([x1_foot, ymin1]))


################################################################################
# Test computing the cylinder radius ###########################################

X0_lowerleft_corner, _  = _project_2D_to_3D(xmax0, ymax0, rvec[0], tvec[0], camera_matrices[0])
X0_lowerright_corner, _  = _project_2D_to_3D(xmin0, ymax0, rvec[0], tvec[0], camera_matrices[0])
#print(X0_lowerleft_corner, X0_lowerright_corner)
radius0 = np.linalg.norm(X0_lowerright_corner[0:2] - X0_lowerleft_corner[0:2]) / 2

X1_lowerleft_corner, _  = _project_2D_to_3D(xmax1, ymax1, rvec[1], tvec[1], camera_matrices[1])
X1_lowerright_corner, _  = _project_2D_to_3D(xmin1, ymax1, rvec[1], tvec[1], camera_matrices[1])
#print(X1_lowerleft_corner, X1_lowerright_corner)
radius1 = np.linalg.norm(X1_lowerright_corner[0:2] - X1_lowerleft_corner[0:2]) / 2

print("radius (cam0, cam1)", radius0, radius1)


################################################################################
# Test `_shift_2D_point_perpendicular` #########################################

dist = radius0

X0_lowerright_corner_temp = _shift_2D_point_perpendicular(P1=X0_foot[0:2], P2=Ctilde0[0:2], dist=dist)
X0_lowerright_corner = np.zeros((3))
X0_lowerright_corner[0:2] = X0_lowerright_corner_temp
#print(X0_foot[0:2], moved_p1)

#print(np.linalg.norm(X0_lowerright_corner_temp - X0_foot[0:2]))
x0_lowerright_corner, _ = project_3D_to_2D(
        X0_lowerright_corner,  # 3D points
        rvec[0].flatten(),  # rotation rvec
        tvec[0].flatten(),  # translation tvec
        camera_matrices
[0],  # camera matrix
        dist_coeffs[0])  # distortion coefficients

print("lower right bbox from cylinder params (projected, real)", x0_lowerright_corner, np.array([xmax0, ymax0]))

# MINUS for the other direction
dist = - radius0

X0_lowerleft_corner_temp = _shift_2D_point_perpendicular(P1=X0_foot[0:2], P2=Ctilde0[0:2], dist=dist)
X0_lowerleft_corner = np.zeros((3))
X0_lowerleft_corner[0:2] = X0_lowerleft_corner_temp
#print(X0_foot[0:2], moved_p1)

#print(np.linalg.norm(X0_lowerleft_corner_temp - X0_foot[0:2]))
x0_lowerleft_corner, _ = project_3D_to_2D(
        X0_lowerleft_corner,  # 3D points
        rvec[0].flatten(),  # rotation rvec
        tvec[0].flatten(),  # translation tvec
        camera_matrices
[0],  # camera matrix
        dist_coeffs[0])  # distortion coefficients

print("lower left bbox from cylinder params (projected, real)", x0_lowerleft_corner, np.array([xmin0, ymax0]))


################################################################################
# test `transform_3D_cylinder_to_2D_bbox_params` ###############################

# TODO: try with best possible cylinder to get impression about errors
bbox0 = transform_3D_cylinder_to_2D_bbox_params(cylinder0, rvec[0], tvec[0], camera_matrices[0], dist_coeffs[0])
bbox1 = transform_3D_cylinder_to_2D_bbox_params(cylinder1, rvec[1], tvec[1], camera_matrices[1], dist_coeffs[1])
bbox2 = transform_3D_cylinder_to_2D_bbox_params(cylinder2, rvec[2], tvec[2], camera_matrices[2], dist_coeffs[2])

print(bbox0, "\n", data[0]["views"][0], "\n", "\n")
print(bbox1, "\n", data[0]["views"][1], "\n", "\n")
print(bbox2, "\n", data[0]["views"][2], "\n", "\n")

################################################################################
