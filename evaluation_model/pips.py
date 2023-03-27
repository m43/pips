import torch
from tqdm import tqdm

from evaluation_model.model import EvaluationModel
from nets.pips import Pips
from pips_utils import saverloader


class PipsEvaluationModel(EvaluationModel):
    DEFAULT_CHECKPOINT_PATH = "reference_model"

    def __init__(self, checkpoint_path, device, stride, s):
        self.device = device
        self.checkpoint_path = checkpoint_path or self.DEFAULT_CHECKPOINT_PATH
        self.stride = stride
        self.s = s

        print(f"Loading PIPS model from {self.checkpoint_path}")
        self.model = Pips(S=s, stride=stride)
        self._loaded_checkpoint_step = saverloader.load(self.checkpoint_path, self.model)
        self.model = self.model.to(device)
        self.model.eval()

    def _forward(self, rgbs, query_points, summary_writer=None):
        batch_size, n_frames, channels, height, width = rgbs.shape
        n_points = query_points.shape[1]

        if not batch_size == 1:
            raise NotImplementedError("Batch size > 1 is not supported for PIPS yet")

        trajectory_list = []
        visibility_list = []
        # TODO: Batchify the for loop for better GPU utilization
        for point_idx in tqdm(range(n_points)):

            trajectory = torch.zeros((n_frames, 2), dtype=torch.float32, device=rgbs.device)
            visibility = torch.zeros((n_frames), dtype=torch.float32, device=rgbs.device)

            start_frame = int(query_points[0, point_idx, 0].item())
            visibility[start_frame] = 1.0
            trajectory[start_frame, :] = query_points[0, point_idx, 1:]

            feat_init = None
            current_frame = start_frame
            while current_frame < n_frames - 1:
                rgbs_input = rgbs[:, current_frame:current_frame + self.model.S, :, :, :]

                last_rgb = rgbs_input[:, -1, :, :, :]
                missing_rgbs = last_rgb.unsqueeze(1).repeat(1, self.model.S - rgbs_input.shape[1], 1, 1, 1)
                rgbs_input = torch.cat([rgbs_input, missing_rgbs], dim=1)

                output_trajectory_per_iteration, _, output_visibility_logits, feat_init, _ = self.model.forward(
                    xys=trajectory[current_frame, :].unsqueeze(0).unsqueeze(0),
                    rgbs=rgbs_input,
                    feat_init=feat_init,
                    iters=6,
                    # sw=summary_writer,  # Slow
                    return_feat=True,
                )
                output_visibility = torch.sigmoid(output_visibility_logits)
                output_trajectory = output_trajectory_per_iteration[-1]

                predicted_frame_range = torch.arange(
                    1,
                    self.model.S - missing_rgbs.shape[1],
                    device=rgbs.device,
                )
                visibility[current_frame + predicted_frame_range] = output_visibility[0, predicted_frame_range, 0]
                trajectory[current_frame + predicted_frame_range, :] = output_trajectory[0, predicted_frame_range, 0, :]

                # TODO Threshold hardcoded
                next_frame_visibility_threshold = 0.9
                next_frame_last_candidate = current_frame + self.model.S - missing_rgbs.shape[1] - 1
                next_frame_earliest_candidate = current_frame + 1
                next_frame = next_frame_last_candidate
                while visibility[next_frame] <= next_frame_visibility_threshold:
                    next_frame -= 1
                    if next_frame < next_frame_earliest_candidate:
                        next_frame_visibility_threshold -= 0.02
                        next_frame = next_frame_last_candidate
                current_frame = next_frame

            trajectory_list += [trajectory]
            visibility_list += [visibility]

        trajectories = torch.stack(trajectory_list, dim=1).unsqueeze(0)
        visibilities = torch.stack(visibility_list, dim=1).unsqueeze(0)
        visibilities = visibilities > 0.5
        return trajectories, visibilities

    def forward(self, rgbs, query_points, summary_writer=None):
        # From left to right
        trajectories_to_right, visibilities_to_right = self._forward(rgbs, query_points, summary_writer)

        # From right to left
        rgbs_flipped = rgbs.flip(1)
        query_points_flipped = query_points.clone()
        query_points_flipped[:, :, 0] = rgbs.shape[1] - query_points_flipped[:, :, 0] - 1
        trajectories_to_left, visibilities_to_left = self._forward(rgbs_flipped, query_points_flipped, summary_writer)
        trajectories_to_left = trajectories_to_left.flip(1)
        visibilities_to_left = visibilities_to_left.flip(1)

        # Merge
        trajectory_list = []
        visibility_list = []
        n_points = query_points.shape[1]
        for point_idx in tqdm(range(n_points)):
            start_frame = int(query_points[0, point_idx, 0].item())

            trajectory = torch.cat([
                trajectories_to_left[0, :start_frame, point_idx, :],
                trajectories_to_right[0, start_frame:, point_idx, :]
            ])
            visibility = torch.cat([
                visibilities_to_left[0, :start_frame, point_idx],
                visibilities_to_right[0, start_frame:, point_idx],
            ])

            assert trajectory.shape == trajectories_to_right[0, :, point_idx, :].shape
            assert visibility.shape == visibilities_to_right[0, :, point_idx].shape

            query_points = query_points.float()
            assert torch.allclose(trajectories_to_right[0, start_frame, point_idx, :], query_points[0, point_idx, 1:])
            assert torch.allclose(trajectories_to_left[0, start_frame, point_idx, :], query_points[0, point_idx, 1:])
            assert torch.allclose(trajectory[start_frame, :], query_points[0, point_idx, 1:])

            assert visibilities_to_right[0, start_frame, point_idx] == 1.0
            assert visibilities_to_left[0, start_frame, point_idx] == 1.0
            assert visibility[start_frame] == 1.0

            trajectory_list += [trajectory]
            visibility_list += [visibility]

        trajectories = torch.stack(trajectory_list, dim=1).unsqueeze(0)
        visibilities = torch.stack(visibility_list, dim=1).unsqueeze(0)
        return trajectories, visibilities
