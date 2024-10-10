# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This file is based on home-robot goat-sim branch

import numpy as np
import skimage.morphology
import torch
import torch.nn as nn


class FrontierExplorationPolicy(nn.Module):
    """
    Frontier exploration: select high-level exploration goals of the closest
    unexplored region.
    """

    def __init__(self) -> None:
        super().__init__()

        self.dilate_explored_kernel = nn.Parameter(
            torch.from_numpy(skimage.morphology.disk(10))
            .unsqueeze(0)
            .unsqueeze(0)
            .float(),
            requires_grad=False,
        )
        self.select_border_kernel = nn.Parameter(
            torch.from_numpy(skimage.morphology.disk(1))
            .unsqueeze(0)
            .unsqueeze(0)
            .float(),
            requires_grad=False,
        )

    @property
    def goal_update_steps(self) -> int:
        return 1

    def forward(self, map_features: np.ndarray) -> np.ndarray:
        """
        Arguments:
            map_features: semantic map features of shape
             (batch_size, 8? + num_sem_categories, M, M) ## TODO

        Returns:
            goal_map: binary map encoding goal(s) of shape (batch_size, M, M)
        """
        # Select unexplored area
        frontier_map = (map_features[:, [MapConstants.EXPLORED_MAP], :, :] == 0).float()

        # Dilate explored area
        frontier_map = binary_erosion(frontier_map, self.dilate_explored_kernel)

        # Select the frontier
        frontier_map = (
            binary_dilation(frontier_map, self.select_border_kernel) - frontier_map
        )
        return frontier_map


# ----------------------------------------- #
# from constants.py
    
class MapConstants:
    NON_SEM_CHANNELS = 4  # Number of non-semantic channels at the start of maps
    OBSTACLE_MAP = 0
    EXPLORED_MAP = 1
    CURRENT_LOCATION = 2
    VISITED_MAP = 3

# ----------------------------------------- #
# from morphology.py
import torch

def binary_dilation(binary_image, kernel):
    """
    Arguments:
        binary_image: binary image tensor of shape (bs, 1, H1, W1)
        kernel: binary structuring element tensor of shape (1, 1, H2, W2)

    Returns:
        binary image tensor of the same shape as input
    """
    return torch.clamp(
        torch.nn.functional.conv2d(binary_image, kernel, padding=kernel.shape[-1] // 2),
        0,
        1,
    )


def binary_erosion(binary_image, kernel):
    """
    Arguments:
        binary_image: binary image tensor of shape (bs, 1, H1, W1)
        kernel: binary structuring element tensor of shape (1, 1, H2, W2)

    Returns:
        binary image tensor of the same shape as input
    """
    return 1 - torch.clamp(
        torch.nn.functional.conv2d(
            1 - binary_image, kernel, padding=kernel.shape[-1] // 2
        ),
        0,
        1,
    )


def binary_opening(binary_image, kernel):
    return binary_dilation(binary_erosion(binary_image, kernel), kernel)


def binary_closing(binary_image, kernel):
    return binary_erosion(binary_dilation(binary_image, kernel), kernel)


def binary_denoising(binary_image, kernel):
    return binary_opening(binary_closing(binary_image, kernel), kernel)
