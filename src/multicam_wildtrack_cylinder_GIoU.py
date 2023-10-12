"""Compute Gernalized Intersection of Union for two Cylinders.

Sketch to potentially replace the bbox GIoU for the case of 3D cylinder
prediction.
"""


import math


import matplotlib.pyplot as plt



def compute_cylinder_GIoU(x1, y1, r1, h1, x2, y2, r2, h2):
    """Extends https://giou.stanford.edu/ to 3D cylinders."""
    max_h = max(h1, h2)

    IoU, Union = _compute_cylinder_IoU(x1, y1, r1, h1, x2, y2, r2, h2)
    C_approx = max_h / 2 * _smallest_enclosing_circle(x1, y1, r1, x2, y2, r2)

    return IoU - (C_approx - Union) / C_approx


def _compute_circle_intersection(x1, y1, r1, x2, y2, r2) -> float:
    """Compute the area of intersection between two circles.
    
    Follows
    https://www.xarg.org/2016/07/calculate-the-intersection-area-of-two-circles/

    """

    # Compute the distance between the centers of the circles
    distance = math.hypot(x2 - x1, y2 - y1)

    if distance < r1 + r2:
        # Compute the squared radius of each circle
        radius1_sq = r1 ** 2
        radius2_sq = r2 ** 2

        # Compute the x-coordinate of the intersection point
        x = (radius1_sq - radius2_sq + distance ** 2) / (2 * distance)

        # Compute the square of the distance from the intersection point to the center of circle 1
        z = x ** 2

        # Compute the y-coordinate of the intersection point
        y = math.sqrt(radius1_sq - z)

        if distance <= abs(r2 - r1):
            # The smaller circle is completely inside the larger circle
            return math.pi * min(radius1_sq, radius2_sq)

        # Compute the area between the circles using the intersection points
        return (radius1_sq * math.asin(y / r1)
                + radius2_sq * math.asin(y / r2)
                - y * (x + math.sqrt(z + radius2_sq - radius1_sq)))

    # The circles do not intersect
    return 0

# Test case from
# https://math.stackexchange.com/questions/402858/area-of-intersection-between-two-circles
# Circle 1
x1 = 0
y1 = 0
radius1 = 3

# Circle 2
x2 = 3
y2 = 0
radius2 = 3

# Compute intersection area and center coordinates
intersection_area = _compute_circle_intersection(x1, y1, radius1, x2, y2, radius2)
print("Intersection Area:", intersection_area)
print(
    intersection_area,
    radius1 ** 2 * (
         (2 * math.pi)/3 - (math.sqrt(3) / 2)
    )
)

def _compute_cylinder_union(x1, y1, r1, h1, x2, y2, r2, h2):
    return (math.pi * (r1 ** 2) * h1
            + math.pi * (r2 ** 2) * h2
            - _compute_circle_intersection(x1, y1, r1, x2, y2, r2)
    )

def _compute_cylinder_IoU(x1, y1, r1, h1, x2, y2, r2, h2):
    return (_compute_circle_intersection(x1, y1, r1, x2, y2, r2)
            / _compute_cylinder_union(x1, y1, r1, h1, x2, y2, r2, h2)
    ), _compute_cylinder_union(x1, y1, r1, h1, x2, y2, r2, h2)




import matplotlib.pyplot as plt

def _smallest_enclosing_circle(x1, y1, r1, x2, y2, r2):
    # Swap circles if necessary
    if r1 > r2:
        x1, y1, r1, x2, y2, r2 = x2, y2, r2, x1, y1, r1

    # Compute the distance between circle centers
    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    # Check if the first circle lies entirely inside the second circle
    if distance + r1 <= r2:
        return x2, y2, r2

    # Compute the center and radius of the smallest enclosing circle
    theta = 0.5 + (r2 - r1) / (2 * distance)
    cx = (1 - theta) * x1 + theta * x2
    cy = (1 - theta) * y1 + theta * y2
    radius = (distance + r1 + r2) / 2

    return cx, cy, radius

def plot_enclosing_circle(x1, y1, r1, x2, y2, r2):
    cx, cy, radius = _smallest_enclosing_circle(x1, y1, r1, x2, y2, r2)

    # Plot the circles
    fig, ax = plt.subplots()
    circle1 = plt.Circle((x1, y1), r1, color='red', alpha=0.5)
    circle2 = plt.Circle((x2, y2), r2, color='blue', alpha=0.5)
    circle3 = plt.Circle((cx, cy), radius, color='green', alpha=0.3)

    ax.set_aspect(1)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)

    # Set plot limits based on circle radii
    max_radius = max(r1, r2, radius)
    ax.set_xlim(cx - max_radius - 1, cx + max_radius + 1)
    ax.set_ylim(cy - max_radius - 1, cy + max_radius + 1)

    # Save the plot as an image file
    plt.savefig('enclosing_circles.png')
    plt.close()

# Example usage
x1, y1, r1 = 0, 0, 1
x2, y2, r2 = 2, 0, 0.5

plot_enclosing_circle(x1, y1, r1, x2, y2, r2)

debug = "db"