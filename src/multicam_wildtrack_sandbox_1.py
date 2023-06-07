"""
Map 2D to 3D bbox coordinates from WIILDTRACK dataset and back following
the approach from Duy's master's thesis.

"""

import numpy as np
from numpy.linalg import inv, pinv
import cv2

from multicam_wildtrack_load_calibration import load_all_intrinsics, load_all_extrinsics


def get_Rt(R, tvec):
    Rt_inhomo = np.concatenate((R, tvec), axis=1)
    #Rt_homo = np.concatenate((Rt_inhomo, np.array([0,0,0,1]).reshape((1,4))), axis=0)
    return Rt_inhomo
    


def project_2d_3d(
        x: float,
        y: float,
        rvec: np.array,
        tvec: np.array,
        K: np.array) -> np.array:
    
    tvec = tvec
    x_homogenous = np.array([x, y, 1], dtype=np.float32).reshape((3,1))


    R, _ = cv2.Rodrigues(rvec)

    Rt = get_Rt(R, tvec)

    P = K @ Rt
    M = K @ R

    Ctilde = - R.T @ tvec

    np.testing.assert_almost_equal(Ctilde,- inv(M) @ P[:,3].reshape((3,1)), decimal=4)

    #Pplus0 = pinv(P)

    Xtilde = inv(M) @ x_homogenous

    Xtilde_pi = - (Ctilde[2] / Xtilde[2]) * (inv(M) @ x_homogenous) + Ctilde

    return Xtilde_pi


rvec, tvec = load_all_extrinsics()


cameraMatrices, distCoeffs = load_all_intrinsics()




# grid to world:
# i is position ID
#        x = _grid_origin[0] + _grid_step * (i % 480)
#        y = _grid_origin[1] + _grid_step * (i / 480)


# 0.025 could be map exapnd (2.5 cm is 1 grid point)
def get_3d_grid_from_positionID(id:int):
    """See WILDTRACK Readme."""
    x_grid = -300 + 2.5 * (id % 480)
    y_grid = -900 + 2.5 * (id / 480)
    return np.float32([[x_grid, y_grid, 0]])

# with i and j instead of i*j:
#origineX + (float) j * width / (float) nb_width, origineY + (float) i * height / (float) nb_height, 0.)

# radius in world coordinates is 0.3



# WILDTRACK has irregular denotion: H*W=480*1440,
# normally x would be \in [0,1440), not [0,480)
# In our data annotation, we follow the regular x \in [0,W),
# and one can calculate x = pos % W, y = pos // W

def get_worldcoord_from_pos(pos):
    grid = get_worldgrid_from_pos(pos)
    return get_worldcoord_from_worldgrid(grid)

def get_worldgrid_from_pos(pos):
    grid_x = pos % (MAP_WIDTH * MAP_EXPAND)
    grid_y = pos // (MAP_WIDTH * MAP_EXPAND)
    return np.array([grid_x, grid_y], dtype=int)

def get_worldcoord_from_worldgrid(worldgrid):
    grid_x, grid_y = worldgrid
    coord_x = grid_x / MAP_EXPAND
    coord_y = grid_y / MAP_EXPAND
    return np.array([coord_x, coord_y])


_grid_sizes = (1440, 480)
_grid_origin = (-300, -900, 0) # with bugfix from toolkit
_grid_step = 2.5


# grid to world:
# i is position ID
#        x = _grid_origin[0] + _grid_step * (i % 480)
#        y = _grid_origin[1] + _grid_step * (i / 480)
"""
{
        "personID": 122,
        "positionID": 456826,
        "views": [
            {
                "viewNum": 0,
                "xmax": 1561,
                "xmin": 1510,
                "ymax": 299,
                "ymin": 139
            },
            {
                "viewNum": 1,
                "xmax": 891,
                "xmin": 813,
                "ymax": 289,
                "ymin": 10
            },
"""

# "personID": 122, "positionID": 456826,
true_3d = get_3d_grid_from_positionID(456826)
true_3d = true_3d


# TODO: Perhaps multiply world coordinates from view projection and put through function?

# position 1510 139 1561 299

#{
#    "viewNum": 0,
#    "xmax": 1561,
#    "xmin": 1510,
#    "ymax": 299,
#    "ymin": 139
#},

xmax0 = 1561
ymax0 = 299

xmin0 = 1510
ymin0 = 139


# get feet position from x center
x0 = (xmax0 + xmin0) / 2 
y0 = ymax0

#print(cameraMatrices[0])
X0  = project_2d_3d(x0, y0, rvec[0], tvec[0], cameraMatrices[0])


x0_test, _ = cv2.projectPoints(
        X0.flatten(),  # 3D points
        rvec[0].flatten(),  # rotation rvec
        tvec[0].flatten(),  # translation tvec
        cameraMatrices[0],  # camera matrix
        distCoeffs[0])  # distortion coefficients


x0_from_true, _ = cv2.projectPoints(
        true_3d,  # 3D points
        rvec[0].flatten(),  # rotation rvec
        tvec[0].flatten(),  # translation tvec
        cameraMatrices[0],  # camera matrix
        distCoeffs[0])  # distortion coefficients
#{
#    "viewNum": 1,
#    "xmax": 891,
#    "xmin": 813,
#    "ymax": 289,
#    "ymin": 10
#},

# position 813 10 891 289
xmax1 = 891
ymax1 = 289

xmin1 = 813
ymin1 = 10

# get feet position from x center
x1 = (xmax1 + xmin1) / 2
y1 = ymax1



X1  = project_2d_3d(x1, y1, rvec[1], tvec[1], cameraMatrices[1])


x1_test, _ = cv2.projectPoints(
        X1.flatten(),  # 3D points
        rvec[1].flatten(),  # rotation rvec
        tvec[1].flatten(),  # translation tvec
        cameraMatrices[1],  # camera matrix
        distCoeffs[1])  # distortion coefficients

x1_from_true, _ = cv2.projectPoints(
        true_3d,  # 3D points
        rvec[1].flatten(),  # rotation rvec
        tvec[1].flatten(),  # translation tvec
        cameraMatrices[1],  # camera matrix
        distCoeffs[1])  # distortion coefficients


print("World coordinates should all be equal...(pos, cam 0->3d, cam1->3d):", true_3d, X0.flatten(), X1.flatten(), "\n")

print("Cam 0 coordinates (pos->2d, 2d->3d->2d, true data):", x0_from_true, x0_test, np.array([x0, y0]), "\n")

print("Cam 1 coordinates (pos->2d, 2d->3d->2d, true data):", x1_from_true, x1_test, np.array([x1, y1]), "\n")

#print(x1_true, np.array([x1, y1]), x1_test, "\n")


# tvec was reduced by 100? step size should perhaps be rather 0.025?
debug_point = ""




























###############################################################################



# https://stackoverflow.com/questions/12977980/in-opencv-converting-2d-image-point-to-3d-world-unit-vector?rq=4




################################################################################
# Camera 1, period 0

### extrinsic
rvec1 = [0.6167870163917542, -2.14595890045166, 1.6577140092849731]
tvec1 = [1195.231201171875, -336.5144958496094, 2040.53955078125]

### intrinsic
# focal lengths expressed in pixel units
fx1 = 1707.266845703125
fy1 = 1719.0408935546875

# principle point / image center
cx1 = 978.1306762695312
cy1 = 417.01922607421875

cam_mat1, rot_trans_mat1 = get_K_and_Rt(rvec1, tvec1, fx1, fy1, cx1, cy1)

# Camera 0 "personID": 122, "positionID": 456826,
xmax0 = 1561
ymax0 = 1510

u1_xmax_t0 = 891
v1_ymax_t0 = 289

homogenous_xy1 = np.array([u1_xmax_t0, v1_ymax_t0 , 1]).reshape((3,1))


################################################################################
debug_point = ""

# Camera 1

###############################################################################
# perhaps mask the pixels that are visible on multile cameras and do MCMOT
# and use rest for MOT
###############################################################################
