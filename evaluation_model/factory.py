from evaluation_model.dino import DinoEvaluationModel
from evaluation_model.model import EvaluationModel
from evaluation_model.pips import PipsEvaluationModel
from evaluation_model.raft import RaftEvaluationModel


class EvaluationModelFactory:
    @classmethod
    def get_model(cls, name, checkpoint_path, device, pips_stride, pips_window) -> "EvaluationModel":
        if name == 'pips':
            return PipsEvaluationModel(checkpoint_path, device, pips_stride, pips_window)
        elif name == 'raft':
            return RaftEvaluationModel(checkpoint_path, device)
        elif name == 'dino':
            return DinoEvaluationModel(checkpoint_path, device)
        else:
            raise ValueError(f"Invalid name given: `{name}`")
