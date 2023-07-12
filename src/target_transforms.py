"""Transformations for different bbox data formats in Trackformer.

Trackformer trains on x_center/W, y_center/H, w/W h/H.
and evaluates on xmin, ymin, xmax, ymax.

It also prevents empty bboxes. I use the respective function in the modules
that generate data.

I need to match the evaluation format and test whether I can train on
x_min/W, y_min/H, w/W h/H.

"""
from wildtrack_globals import W, H
import torch


from wildtrack_globals import X_CENTER, Y_CENTER, HEIGHT, RADIUS

MIN_VALS = [X_CENTER["min"], Y_CENTER["min"], HEIGHT["min"], RADIUS["min"]]
MAX_VALS = [X_CENTER["max"], Y_CENTER["max"], HEIGHT["max"], RADIUS["max"]]


def min_max_scaling(cylinders, min_vals=MIN_VALS, max_vals=MAX_VALS):
    """
    Apply min-max scaling to a torch.Tensor along the vertical axis with different
    min and max values for each column.

    Args:
        cylinder (torch.Tensor): Input tensor of shape (N, 4).
        min_vals (list or tuple): List or tuple of length 4 containing the minimum values
                                  for each column.
        max_vals (list or tuple): List or tuple of length 4 containing the maximum values
                                  for each column.

    Returns:
        torch.Tensor: Scaled tensor of the same shape as the input tensor.
    """
    # Perform min-max scaling
    scaled_tensor = (cylinders - torch.tensor(min_vals, dtype=cylinders.dtype, device=cylinders.device)) \
                    / (torch.tensor(max_vals, dtype=cylinders.dtype, device=cylinders.device) \
                       - torch.tensor(min_vals, dtype=cylinders.dtype, device=cylinders.device))

    return scaled_tensor


def inverse_min_max_scaling(scaled_tensor, min_vals=MIN_VALS, max_vals=MAX_VALS):
    original_tensor = scaled_tensor * (torch.tensor(max_vals, dtype=scaled_tensor.dtype, device=scaled_tensor.device) \
                                       - torch.tensor(min_vals, dtype=scaled_tensor.dtype, device=scaled_tensor.device)) \
                      + torch.tensor(min_vals, dtype=scaled_tensor.dtype, device=scaled_tensor.device)

    return original_tensor

"""
# test
# Input tensor
tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [9.0, 10.0, 11.0, 12.0]])

# Minimum and maximum values for each column
min_vals_ = [0.0, 1.0, 2.0, 3.0]
max_vals_ = [10.0, 20.0, 30.0, 40.0]

# Apply min-max scaling
scaled_tensor = min_max_scaling(tensor, min_vals_, max_vals_)
original_tensor = inverse_min_max_scaling(scaled_tensor, min_vals_, max_vals_)

print(tensor, scaled_tensor, original_tensor)
"""


def prevent_empty_bboxes(boxes: torch.Tensor):
    boxes = bbox_xywh_to_xyxy(boxes)
    boxes[:, 0::2].clamp_(min=0, max=W)
    boxes[:, 1::2].clamp_(min=0, max=H)
    boxes = bbox_xyxy_to_wywh(boxes)
    return boxes


def clamp_x(x: float):
    min_value = 0
    max_value = W
    return max(min_value, min(x, max_value))


def clamp_y(y: float):
    min_value = 0
    max_value = H
    return max(min_value, min(y, max_value))


def bbox_xywh_to_xyxy(x):
    xmin, ymin, w, h = x.unbind(-1)
    # boxes[:, 2:] += boxes[:, :2]
    b = [(xmin), (ymin),
         (xmin + w), (ymin + h)]
    return torch.stack(b, dim=-1)


#-------------------------------------------------------------------------------
# Unused copies from trackformer/util/box_ops.py

def bbox_xyxy_to_wywh(boxes: torch.Tensor):
    boxes[:, 2:] -= boxes[:, :2]
    return boxes


def bbox_xyxy_to_cxcywh(boxes: torch.Tensor):
    """2nd target transform."""
    # 1st target transform in transforms.py Normalize
    x0, y0, x1, y1 = boxes.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def unit_cube_bbox(boxes: torch.Tensor):
    """3rd target transform."""
    # 2nd target transform in transforms.py Normalize
    return boxes / torch.tensor([W, H, W, H], dtype=torch.float32).to(device=boxes.device)


def inv_unit_cube_bbox(boxes: torch.Tensor):
    """Inverse of `unit_cube_bbox`."""
    # 2nd target transform in transforms.py Normalize
    return boxes * torch.tensor([W, H, W, H], dtype=torch.float32).to(device=boxes.device)


def deformable_DETR_postprocess(boxes: torch.Tensor):
    """Inverse of stacked transforms 1 to 3 from above.
    
    I.e. (x_center/W. y_center/H, w/W. h/H() -> (xmin, ymin, w, h)

    """
    # convert to [x0, y0, x1, y1] format
    boxes = _box_cxcywh_to_xyxy(boxes)
    # and from relative [0, 1] to absolute [0, height] coordinates
    scale_fct = torch.stack([W, H, W, H], dim=1).to(device=boxes.device)
    boxes = boxes * scale_fct[:, None, :]


def _box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

#-------------------------------------------------------------------------------