from abc import ABC, abstractmethod
from typing import Tuple

import torch


class EvaluationModel(ABC):
    """
    Abstract class for evaluation models.
    """

    @abstractmethod
    def forward(self, rgbs, query_points, summary_writer=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the model and returns the predicted trajectories and visibilities.

        :param rgbs: A tensor of shape (batch_size, n_frames, channels, height, width) containing the RGB images.
        :param query_points: A tensor of shape (batch_size, n_points, 3) containing the query points,
            each point being (t, x, y).
        :param summary_writer: A pips_utils.improc.Summ_writer object that the model can use to write summaries.
        :return: A tuple containing:
            - A tensor of shape (batch_size, n_frames, n_points, 2) containing the predicted trajectories.
            - A tensor of shape (batch_size, n_frames, n_points) containing the predicted visibilities.
        """
        pass

    def evaluate_batch(self, trajectories_gt, visibilities_gt, rgbs, query_points, summary_writer):
        """
        Evaluates a batch of data and returns the results.
        :param trajectories_gt: A 4D tensor representing the ground-truth trajectory.
            Its shape is (batch_size, n_frames, n_points, 2).
        :param visibilities_gt: A 3D tensor representing the ground-truth visibilities.
            Its shape is (batch_size, n_frames, n_points).
        :param rgbs: A tensor of shape (batch_size, n_frames, channels, height, width) containing the RGB images.
        :param query_points: A tensor of shape (batch_size, n_points, 3) containing the query points,
            each point being (t, x, y).
        :param summary_writer: A pips_utils.improc.Summ_writer object that the model can use to write summaries.
        :return: A dictionary containing the results.
        """
        trajectories_pred, visibilities_pred = self.forward(rgbs, query_points, summary_writer)
        batch_size = rgbs.shape[0]
        n_frames = rgbs.shape[1]
        n_points = trajectories_gt.shape[2]
        assert trajectories_pred.shape == (batch_size, n_frames, n_points, 2)

        results = {
            "trajectories_gt": trajectories_gt.detach().clone().cpu(),
            "visibilities_gt": visibilities_gt.detach().clone().cpu(),
            "trajectories_pred": trajectories_pred.detach().clone().cpu(),
            "visibilities_pred": visibilities_pred.detach().clone().cpu(),
            "query_points": query_points.detach().clone().cpu(),
        }

        return results

    @classmethod
    def unpack_results(cls, packed_results, batch_idx):
        unpacked_results_list = []
        for b in range(packed_results["trajectories_gt"].shape[0]):
            for n in range(packed_results["trajectories_gt"].shape[2]):
                result = {
                    "iter": batch_idx,
                    "video_idx": b,
                    "point_idx_in_video": n,
                    "trajectory_gt": packed_results["trajectories_gt"][b, :, n, :],
                    "trajectory_pred": packed_results["trajectories_pred"][b, :, n, :],
                    "visibility_gt": packed_results["visibilities_gt"][b, :, n],
                    "visibility_pred": packed_results["visibilities_pred"][b, :, n],
                    "query_point": packed_results["query_points"][b, n, :],
                }
                unpacked_results_list += [result]
        return unpacked_results_list
