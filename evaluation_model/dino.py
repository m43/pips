import torch

import pips_utils.test
from evaluation_model.model import EvaluationModel


class DinoEvaluationModel(EvaluationModel):
    DEFAULT_CHECKPOINT = ('facebookresearch/dino:main', 'dino_vits8')

    def __init__(self, checkpoint, device):
        self.checkpoint = checkpoint or self.DEFAULT_CHECKPOINT
        repo_name, model_name = self.checkpoint
        print(f"Loading DINO model from repo={repo_name} model={model_name}")
        self.model = torch.hub.load(repo_name, model_name, pretrained=True).to(device)
        self.model.eval()

    def forward(self, rgbs, query_points, summary_writer=None):
        raise NotImplementedError()  # TODO
        batch_size, n_frames, channels, height, width = rgbs.shape
        n_points = query_points.shape[1]
        trajectories = pips_utils.test.get_dino_output(model, rgbs, trajectories_gt, visibilities_gt)
        visibilities = (trajectories[:, :, :, 0] >= 0) & \
                       (trajectories[:, :, :, 1] >= 0) & \
                       (trajectories[:, :, :, 0] < width) & \
                       (trajectories[:, :, :, 1] < height)
        return trajectories, visibilities
