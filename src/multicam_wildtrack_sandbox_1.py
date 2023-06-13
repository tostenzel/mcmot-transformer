"""
Map 2D to 3D bbox coordinates from WIILDTRACK dataset and back following
the approach from Duy's master's thesis.

```
# 3D world grid paramters from official WILDTRACK toolkit
_grid_sizes = (1440, 480)
_grid_origin = (-300, -900, 0) # with bugfix from toolkit
_grid_step = 2.5
```

"""
import math

import cv2
import numpy as np
from numpy.linalg import inv, pinv
from scipy.optimize import minimize

from multicam_wildtrack_load_calibration import load_all_intrinsics, load_all_extrinsics


MAN_RADIUS_CM = 16
MAN_HEIGHT_CM = 180


def project_2d_to_3d(
        x: float,
        y: float,
        rvec: np.array,
        tvec: np.array,
        K: np.array
) -> np.array:
    """Project 2d image plain point to 3d world grid.
    
    Requres extrinsic and intrinsic parameters.
    """
    
    #def get_Rt(R, tvec):
    #    return np.concatenate((R, tvec), axis=1)

    x_homogenous = np.array([x, y, 1], dtype=np.float32).reshape((3,1))
    R, _ = cv2.Rodrigues(rvec)
    #Rt = get_Rt(R, tvec)
    #P = K @ Rt
    M = K @ R

    # Ctilde coordinates of the camera centre C in the world coordinate 
    Ctilde = - R.T @ tvec
    #np.testing.assert_almost_equal(Ctilde,- inv(M) @ P[:,3].reshape((3,1)), decimal=4)
    #Pplus0 = pinv(P)
    Xtilde = inv(M) @ x_homogenous
    Xtilde_pi = - (Ctilde[2] / Xtilde[2]) * (inv(M) @ x_homogenous) + Ctilde
  
    return Xtilde_pi, Ctilde


# load list of rvec and tvecs and camera matrices
rvec, tvec = load_all_extrinsics()
cameraMatrices, distCoeffs = load_all_intrinsics()


def get_3d_grid_coordinates_from_positionID(id:int):
    """Decode WILDTRACK position ID into 3d world grid foot point.
    
    Taken from Wildtrack Readme but with intercept and coefficient times scale := 100.
    2.5cm is one grid point step, 480 is image height.
    """
    x_grid = -300 + 2.5 * (id % 480)
    y_grid = -900 + 2.5 * (id / 480)
    return np.float32([[x_grid, y_grid, 0]])

# radius in world coordinates is 0.3
# Note that the origin of the image coordinate system is the upper left corner!

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
true_3d = get_3d_grid_coordinates_from_positionID(456826)
true_3d = true_3d

########################################################################################################################
# cam 0 ################################################################################################################
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
X0, _  = project_2d_to_3d(x0, y0, rvec[0], tvec[0], cameraMatrices[0])


########################################################################################################################
# get height

def objective_function(z4, P1, P2, P3):
    """
    Compute distance between 3D point and line between two 3d points.

    More specifically, the 3D point P4=(x2, y2, z4) is above P2=(x2,y2,z2=0) on the
    ground plane and the line is between P1=(x1,y1,z1) and and P3=(x3,y3,z3=0),
    also on the ground plane
    """

    z4 = z4[0]
    # Calculate the direction vector of the line
    direction_vector = np.array([P3[0] - P1[0], P3[1] - P1[1], P3[2] - P1[2]])

    # Calculate the magnitude of the direction vector
    magnitude = np.linalg.norm(direction_vector)

    # Normalize the direction vector
    unit_direction_vector = direction_vector / magnitude

    # Calculate the point P4
    P4 = np.array([P2[0], P2[1], z4])

    # Calculate the vector from P1 to P4
    vector_P1_P4 = P4 - P1

    # Calculate the distance between the line and the point P2
    distance = np.linalg.norm(np.cross(unit_direction_vector, vector_P1_P4))

    return distance


X0_head_floor, Ctilde  = project_2d_to_3d(x0, ymin0, rvec[0], tvec[0], cameraMatrices[0])

# Example usage
P1 = Ctilde.flatten()
P2 = X0.flatten()
P3 = X0_head_floor.flatten()

# Define the objective function with P1, P2, P3 as additional arguments
objective = lambda z4: objective_function(z4, P1, P2, P3)

# Minimize the objective function starting from an initial guess
initial_guess = 180/2.5 # 1.80m in 3d grid units
result = minimize(objective, initial_guess)

# Get the optimized value of z4
z4_optimized = result.x[0]

# Create the final point P4

X0_head = np.copy(X0)
X0_head[2,0] = z4_optimized
x0_head_test, _ = cv2.projectPoints(
        X0_head.flatten(),  # 3D points
        rvec[0].flatten(),  # rotation rvec
        tvec[0].flatten(),  # translation tvec
        cameraMatrices[0],  # camera matrix
        distCoeffs[0])  # distortion coefficients

#print(x0_head_test, np.array([x0, ymin0]))


########################################################################################################################
# get cylinder radius ##################################################################################################


X0_left, _  = project_2d_to_3d(xmin0, y0, rvec[0], tvec[0], cameraMatrices[0])
X0_right, _  = project_2d_to_3d(xmax0, y0, rvec[0], tvec[0], cameraMatrices[0])

print(X0_left.flatten(), X0_right.flatten())

radius = np.linalg.norm(X0_left.flatten()[0:2] - X0_right.flatten()[0:2]) / 2

########################################################################################################################
# select most exterior point on cylinder as seen by one camera and project it to the respective image plane ############

def move_2d_point_by_distance(x1, x2, z):

    # Step 1: Calculate the direction vector
    dx = x2[0] - x1[0]
    dy = x2[1] - x1[1]
    
    # Step 2: Normalize the direction vector
    magnitude = math.sqrt(dx ** 2 + dy ** 2)
    normalized_dx = dx / magnitude
    normalized_dy = dy / magnitude
    
    # Step 3: Scale the normalized direction vector by z units
    displacement_x = normalized_dx * z
    displacement_y = normalized_dy * z
    
    # Step 4: Add the displacement vector to x1 to get the new position
    new_x1 = np.array([x1[0] + displacement_x, x1[1] + displacement_y]).reshape((2,1))
    
    return new_x1


# requires radius
x1 = X0_right.flatten()[0:2]
x2 = Ctilde.flatten()[0:2]
z = radius

moved_x1 = move_2d_point_by_distance(x1, x2, z)
X0_right = np.zeros((3,1))
X0_right[0:2] = moved_x1
#print(x1, moved_x1)

x0_right, _ = cv2.projectPoints(
        X0_right.flatten(),  # 3D points
        rvec[0].flatten(),  # rotation rvec
        tvec[0].flatten(),  # translation tvec
        cameraMatrices[0],  # camera matrix
        distCoeffs[0])  # distortion coefficients

print(x0_right, np.array([xmax0, ymax0]))




# requires radius
x1 = X0_left.flatten()[0:2]
x2 = Ctilde.flatten()[0:2]
# negative for left corner
z = -radius

moved_x1 = move_2d_point_by_distance(x1, x2, z)
X0_left = np.zeros((3,1))
X0_left[0:2] = moved_x1
#print(x1, moved_x1)

x0_left, _ = cv2.projectPoints(
        X0_left.flatten(),  # 3D points
        rvec[0].flatten(),  # rotation rvec
        tvec[0].flatten(),  # translation tvec
        cameraMatrices[0],  # camera matrix
        distCoeffs[0])  # distortion coefficients

print(x0_left, np.array([xmin0, ymax0]))








########################################################################################################################


x0_test, _ = cv2.projectPoints(
        X0.flatten(),  # 3D points
        rvec[0].flatten(),  # rotation rvec
        tvec[0].flatten(),  # translation tvec
        cameraMatrices[0],  # camera matrix
        distCoeffs[0])  # distortion coefficients

print(x0_test, np.array([x0, ymax0]))


x0_from_true, _ = cv2.projectPoints(
        true_3d,  # 3D points
        rvec[0].flatten(),  # rotation rvec
        tvec[0].flatten(),  # translation tvec
        cameraMatrices[0],  # camera matrix
        distCoeffs[0])  # distortion coefficients

print(x0_from_true, np.array([x0, ymax0]))

########################################################################################################################
# cam 1 ################################################################################################################

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



X1, _  = project_2d_to_3d(x1, y1, rvec[1], tvec[1], cameraMatrices[1])


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

########################################################################################################################

print("World coordinates should all be equal...(pos, cam 0->3d, cam1->3d):", true_3d, X0.flatten(), X1.flatten(), "\n")

print("Cam 0 coordinates (pos->2d, 2d->3d->2d, true data):", x0_from_true, x0_test, np.array([x0, y0]), "\n")

print("Cam 1 coordinates (pos->2d, 2d->3d->2d, true data):", x1_from_true, x1_test, np.array([x1, y1]), "\n")


"""

# Define the two vectors
vector_a = np.array([[x1, y1, z1], [x2, y2, z2]])  # Replace x1, y1, z1, x2, y2, z2 with actual values
vector_b = np.array([[x3, y3, z3], [x4, y4, z4]])  # Replace x3, y3, z3, x4, y4, z4 with actual values

"""





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
