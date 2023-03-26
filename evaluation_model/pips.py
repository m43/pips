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
        raise NotImplementedError()  # TODO
        model: Pips = model
        preds, preds_anim, vis_e, stats = model(
            xys=trajectories_gt[:, 0],
            rgbs=rgbs,
            iters=6,
            trajs_g=trajectories_gt,
            vis_g=visibilities_gt,
            sw=summary_writer,
        )
        return preds[-1]
