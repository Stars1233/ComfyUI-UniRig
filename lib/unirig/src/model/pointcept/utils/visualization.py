"""
Visualization Utils

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
import torch
import trimesh


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.clone().detach().cpu().numpy()
    assert isinstance(x, np.ndarray)
    return x


def save_point_cloud(coord, color=None, file_path="pc.ply", logger=None):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    coord = to_numpy(coord)
    if color is not None:
        color = to_numpy(color)
        # Convert color from [0,1] float to [0,255] uint8 if needed
        if color.max() <= 1.0:
            color = (color * 255).astype(np.uint8)
    pc = trimesh.PointCloud(vertices=coord, colors=color)
    pc.export(file_path)
    if logger is not None:
        logger.info(f"Save Point Cloud to: {file_path}")


def save_bounding_boxes(
    bboxes_corners, color=(1.0, 0.0, 0.0), file_path="bbox.ply", logger=None
):
    """
    Save bounding boxes as PLY.

    NOTE: Line set export is not supported without open3d.
    This is a visualization-only function not used in inference.
    """
    raise NotImplementedError(
        "save_bounding_boxes requires open3d LineSet which has been removed. "
        "This visualization function is not used in inference."
    )


def save_lines(
    points, lines, color=(1.0, 0.0, 0.0), file_path="lines.ply", logger=None
):
    """
    Save lines as PLY.

    NOTE: Line set export is not supported without open3d.
    This is a visualization-only function not used in inference.
    """
    raise NotImplementedError(
        "save_lines requires open3d LineSet which has been removed. "
        "This visualization function is not used in inference."
    )
