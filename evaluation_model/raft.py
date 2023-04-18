import torch

import pips_utils.improc
import pips_utils.samp
from evaluation_model.model import EvaluationModel
from nets.raftnet import Raftnet


class RaftEvaluationModel(EvaluationModel):
    DEFAULT_CHECKPOINT_PATH = "raft_ckpts/raft-things.pth"

    def __init__(self, checkpoint_path, device):
        self.checkpoint_path = checkpoint_path or self.DEFAULT_CHECKPOINT_PATH
        print(f"Loading Raft model from {self.checkpoint_path}")
        self.model = Raftnet(ckpt_name=self.checkpoint_path).to(device)
        self.model.eval()

    def forward(self, rgbs, query_points, summary_writer=None):
        batch_size, n_frames, channels, height, width = rgbs.shape
        n_points = query_points.shape[1]

        prep_rgbs = pips_utils.improc.preprocess_color(rgbs)

        flows_forward = []
        flows_backward = []
        for t in range(1, n_frames):
            rgb0 = prep_rgbs[:, t - 1]
            rgb1 = prep_rgbs[:, t]
            flows_forward.append(self.model.forward(rgb0, rgb1, iters=32)[0])
            flows_backward.append(self.model.forward(rgb1, rgb0, iters=32)[0])
        flows_forward = torch.stack(flows_forward, dim=1)
        flows_backward = torch.stack(flows_backward, dim=1)
        assert flows_forward.shape == flows_backward.shape == (batch_size, n_frames - 1, 2, height, width)

        coords = []
        for t in range(n_frames):
            if t == 0:
                coord = torch.zeros_like(query_points[:, :, 1:])
            else:
                prev_coord = coords[t - 1]
                delta = pips_utils.samp.bilinear_sample2d(
                    im=flows_forward[:, t - 1],
                    x=prev_coord[:, :, 0],
                    y=prev_coord[:, :, 1],
                ).permute(0, 2, 1)
                assert delta.shape == (batch_size, n_points, 2), "Forward flow at the discrete points"
                coord = prev_coord + delta

            # Set the ground truth query point location if the timestep is correct
            query_point_mask = query_points[:, :, 0] == t
            coord = coord * ~query_point_mask.unsqueeze(-1) + query_points[:, :, 1:] * query_point_mask.unsqueeze(-1)

            coords.append(coord)

        for t in range(n_frames - 2, -1, -1):
            coord = coords[t]
            successor_coord = coords[t + 1]

            delta = pips_utils.samp.bilinear_sample2d(
                im=flows_backward[:, t],
                x=successor_coord[:, :, 0],
                y=successor_coord[:, :, 1],
            ).permute(0, 2, 1)
            assert delta.shape == (batch_size, n_points, 2), "Backward flow at the discrete points"

            # Update only the points that are located prior to the query point
            prior_to_query_point_mask = t < query_points[:, :, 0]
            coord = (coord * ~prior_to_query_point_mask.unsqueeze(-1) +
                     (successor_coord + delta) * prior_to_query_point_mask.unsqueeze(-1))
            coords[t] = coord

        trajectories = torch.stack(coords, dim=1)
        visibilities = (trajectories[:, :, :, 0] >= 0) & \
                       (trajectories[:, :, :, 1] >= 0) & \
                       (trajectories[:, :, :, 0] < width) & \
                       (trajectories[:, :, :, 1] < height)
        return trajectories, visibilities
