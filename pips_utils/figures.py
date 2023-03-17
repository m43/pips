import os
from typing import Dict, List

import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
sns.set_style("ticks")
sns.set_palette("flare")

plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
plt.rc('text', usetex=True)
plt.rcParams.update({
    # 'figure.figsize': (15, 5),
    'figure.titlesize': '20',
    'axes.titlesize': '22',
    'legend.title_fontsize': '16',
    'legend.fontsize': '14',
    'axes.labelsize': '18',
    'xtick.labelsize': '16',
    'ytick.labelsize': '16',
    'figure.dpi': 200,
})

palette = ["GoldenRod", "r", "forestgreen", "yellow"]


def average_displacement_error(trajectory_a: torch.Tensor, trajectory_b: torch.Tensor) -> float:
    """
    Computes the average displacement error between two trajectory tensors.

    Parameters
    ----------
    trajectory_a : torch.Tensor
        A 2D tensor representing the first trajectory. Its shape should be (S, D),
        where S is the number of time steps and D is the number of dimensions.
    trajectory_b : torch.Tensor
        A 2D tensor representing the second trajectory. Its shape should be (S, D),
        where S is the number of time steps and D is the number of dimensions.

    Returns
    -------
    float
        The average displacement error between the two trajectories, computed as the
        mean L2 norm of the element-wise difference between the two trajectories.

    Raises
    ------
    AssertionError
        If either of the input tensors is not a 2D tensor, or if they do not have
        the same shape.

    Examples
    --------
    >>> trajectory_a = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    >>> trajectory_b = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    >>> average_displacement_error(trajectory_a, trajectory_b)
    1.4142135381698608
    """
    assert trajectory_a.ndim == trajectory_b.ndim == 2, "Input tensors must be 2D tensors"
    assert trajectory_a.shape == trajectory_b.shape, "Input tensors must have the same shape"
    return (trajectory_a - trajectory_b).norm(dim=1).mean().item()


def extract_visible_trajectory(trajectory_a: torch.Tensor, trajectory_b: torch.Tensor,
                               visibility: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts the visible portion of two trajectory tensors according to a visibility mask.

    Parameters
    ----------
    trajectory_a : torch.Tensor
        A 2D tensor representing the first trajectory. Its shape should be (S, D),
        where S is the number of time steps and D is the number of dimensions.
    trajectory_b : torch.Tensor
        A 2D tensor representing the second trajectory. Its shape should be (S, D),
        where S is the number of time steps and D is the number of dimensions.
    visibility : torch.Tensor
        A 1D tensor representing the visibility of each time step. Its length should be S.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple of two 2D tensors representing the visible portion of the input trajectories.
        The output tensor shapes are (N, D), where N is the number of visible time steps
        and D is the number of dimensions.

    Examples
    --------
    >>> trajectory_a = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    >>> trajectory_b = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    >>> visibility = torch.tensor([1, 0, 1])
    >>> extract_visible_trajectory_chain(trajectory_a, trajectory_b, visibility)
    (tensor([[0., 0.], [2., 2.]]), tensor([[1., 1.], [3., 3.]]))
    """
    return trajectory_a[visibility == 1], trajectory_b[visibility == 1]


def extract_visible_trajectory_chain(trajectory_a: torch.Tensor, trajectory_b: torch.Tensor,
                                     visibility: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts the visible portion of two trajectory tensors according to a visibility mask.

    Parameters
    ----------
    trajectory_a : torch.Tensor
        A 2D tensor representing the first trajectory. Its shape should be (S, D),
        where S is the number of time steps and D is the number of dimensions.
    trajectory_b : torch.Tensor
        A 2D tensor representing the second trajectory. Its shape should be (S, D),
        where S is the number of time steps and D is the number of dimensions.
    visibility : torch.Tensor
        A 1D tensor representing the visibility of each time step. Its length should be S.

    Returns
    -------
    tuple
        A tuple of two 2D tensors representing the visible portion of the input trajectories.
        If the entire trajectories are visible, this is just the original input trajectories.

    Raises
    ------
    AssertionError
        If the visibility tensor is not a 1D tensor, or if it does not have the same length as
        the input trajectories.
        If the visibility tensor indicates that the first time step is occluded.

    Examples
    --------
    >>> trajectory_a = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    >>> trajectory_b = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    >>> visibility = torch.tensor([1, 1, 0])
    >>> extract_visible_trajectory_chain(trajectory_a, trajectory_b, visibility)
    (tensor([[0., 0.], [1., 1.]]), tensor([[1., 1.], [2., 2.]]))
    """
    assert visibility.ndim == 1, "Visibility tensor must be a 1D tensor"
    assert len(visibility) == len(trajectory_a) == len(trajectory_b), "Input tensors must have the same length"
    occluded_indices = (visibility == 0).nonzero()
    if len(occluded_indices) > 0:
        first_occluded_index = occluded_indices[0].item()
        assert first_occluded_index != 0, "The first index must not be occluded"
        trajectory_a = trajectory_a[:first_occluded_index]
        trajectory_b = trajectory_b[:first_occluded_index]
    return trajectory_a, trajectory_b


def compute_summary(results: Dict) -> Dict:
    """
    Computes a summary of the trajectory prediction results.

    Parameters
    ----------
    results : Dict
        A dictionary containing the trajectory prediction results. It should have the following keys:
        - 'valids': A 1D boolean tensor indicating which points in the trajectory are valid.
        - 'trajectory_gt': A 2D tensor representing the ground-truth trajectory. Its shape should be (S, D),
          where S is the number of time steps and D is the number of dimensions.
        - 'trajectory_pred': A 2D tensor representing the predicted trajectory. Its shape should be (S, D),
          where S is the number of time steps and D is the number of dimensions.
        - 'visibility_gt': A 1D tensor representing the visibility of each time step. Its length should be S.

    Returns
    -------
    Dict
        A dictionary containing the computed summary statistics, with the following keys:
        - 'idx': A string representing the trajectory index, in the format "<iter>--<video_idx>--<point_idx_in_video>".
        - 'ade': The average displacement error between the ground-truth and predicted trajectories.
        - 'ade_visible': The average displacement error between the visible portion
           of the ground-truth and predicted trajectories.
        - 'ade_visible_chain': The average displacement error between the first visible chain
           of the ground-truth and predicted trajectories.
        - 'n_timesteps': The length of the trajectory.
        - 'n_visible': The number of visible points in the ground-truth trajectory.
        - 'n_visible_chain': The number of visible points in the ground-truth trajectory, assuming a chain structure.

    Examples
    --------
    >>> results = {
    ...     'iter': 123,
    ...     'video_idx': 2,
    ...     'point_idx_in_video': 31,
    ...     'valids': torch.tensor([True, True, True]),
    ...     'trajectory_gt': torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
    ...     'trajectory_pred': torch.tensor([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0]]),
    ...     'visibility_gt': torch.tensor([1, 1, 0])
    ... }
    >>> compute_summary(results)
    {
        'idx': '123--2--31',
        'ade': 0.9428090453147888,
        'ade_visible': 0.7071067690849304,
        'ade_visible_chain': 0.7071067690849304,
        'n_timesteps': 3,
        'n_visible': 2,
        'n_visible_chain': 2
    }
    """
    assert results["valids"].all(), "Assume all points are valid for each timestep"
    traj_gt = results["trajectory_gt"]
    traj_pred = results["trajectory_pred"]
    vis_gt = results["visibility_gt"]
    traj_gt_visible, traj_pred_visible = extract_visible_trajectory(traj_gt, traj_pred, vis_gt)
    traj_gt_visible_chain, traj_pred_visible_chain = extract_visible_trajectory_chain(traj_gt, traj_pred, vis_gt)
    assert vis_gt.sum() == len(traj_gt_visible)
    assert vis_gt.sum() >= len(traj_gt_visible_chain)
    summary = {
        "idx": f'{results["iter"]}--{results["video_idx"]}--{results["point_idx_in_video"]}',
        "ade": average_displacement_error(traj_gt, traj_pred),
        "ade_visible": average_displacement_error(traj_gt_visible, traj_pred_visible),
        "ade_visible_chain": average_displacement_error(traj_gt_visible_chain, traj_pred_visible_chain),
        "n_timesteps": len(traj_gt),
        "n_visible": len(traj_gt_visible),
        "n_visible_chain": len(traj_gt_visible_chain),
    }
    return summary


def compute_summary_df(results_list: List[Dict]) -> pd.DataFrame:
    """
    Computes a summary dataframe of the results of multiple trajectory prediction experiments.

    Parameters:
        results_list (List[Dict]): a list of dictionaries, where each dictionary has the same format as the dictionary
                                   returned by the compute_summary function.

    Returns:
        summary_df (pd.DataFrame): a dataframe containing the following columns:
            - 'idx' (str): a unique identifier for each trajectory.
            - 'ade' (float): the average displacement error between the ground truth and predicted trajectories.
            - 'ade_visible' (float): the average displacement error between the visible parts
                                     of the ground truth and predicted trajectories.
            - 'ade_visible_chain' (float): the average displacement error between the visible trajectory chains
                                           of the ground truth and predicted trajectories.
            - 'n_timesteps' (int): the length of the trajectory.
            - 'n_visible' (int): the number of visible points in the ground truth trajectory.
            - 'n_visible_chain' (int): the number of visible points in the visible trajectory chain
                                       of the ground truth trajectory.
    """
    summaries = []
    for results in results_list:
        summaries += [compute_summary(results)]
    return pd.DataFrame.from_records(summaries)

