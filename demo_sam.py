import argparse
import glob
import os

import cv2
import numpy as np
import torch.cuda
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from tqdm import tqdm

from pips_utils.util import setup_wandb, log_video_to_wandb


def main(args):
    input_video = load_video(args.input_video_folder)
    sam_generator = create_sam_generator(args.checkpoint_model_name, args.checkpoint_path)
    setup_wandb(args.wandb_entity, args.wandb_project, args.experiment_name)
    sam_predictions = get_predictions(input_video, sam_generator)
    output_video = add_masks_to_frames(input_video, sam_predictions)
    output_and_input_video = [np.concatenate([i, o], axis=1) for i, o in zip(input_video, output_video, strict=True)]
    log_video_to_wandb("input", input_video)
    log_video_to_wandb("sam_output", output_video)
    log_video_to_wandb("sam_output_and_input", output_and_input_video)


def load_video(frames_path):
    frame_paths = glob.glob(os.path.join(frames_path, "*.*"))
    frame_paths = sorted(frame_paths)
    print(f"Video frames to be loaded: {frame_paths}")
    frames = [cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB) for frame_path in frame_paths]
    return frames


def create_sam_generator(model_name, checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_name](checkpoint_path)
    sam.to(device)
    return SamAutomaticMaskGenerator(sam)


def get_predictions(frames, sam_generator: SamAutomaticMaskGenerator):
    mask_records_list = [sam_generator.generate(frame) for frame in tqdm(frames)]
    return mask_records_list


def add_masks_to_frames(frames, mask_records_list):
    max_records = max(len(mask_records) for mask_records in mask_records_list)
    color_palette = [np.random.random((1, 1, 3)) for _ in range(max_records)]
    frames_with_masks = []
    for frame, mask_records in tqdm(zip(frames, mask_records_list, strict=True)):
        frame = frame / 255
        mask_records = sorted(mask_records, key=lambda x: x["area"], reverse=True)
        for i, mask_record in enumerate(mask_records):
            mask = mask_record["segmentation"]
            overlay = np.ones((mask.shape[0], mask.shape[1], 3))
            overlay *= color_palette[i]
            overlay = np.where(mask[..., None] > 0, overlay, frame)
            frame = cv2.addWeighted(frame, 0.2, overlay, 0.8, 0)
        frame = frame * 255
        frame = frame.astype(np.uint8)
        frames_with_masks += [frame]
    return frames_with_masks


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video_folder", type=str, default="demo_images/black_french_bulldog",
                        help="Path to folder with images of video frames.")
    parser.add_argument("--checkpoint_path", type=str, default="sam_checkpoints/sam_vit_h_4b8939.pth",
                        help="Path to SAM model checkpoint.")
    parser.add_argument("--checkpoint_model_name", type=str, default="vit_h",
                        help="Name of the model from the SAM registry.")
    parser.add_argument('--wandb_entity', type=str, default='point-tracking')
    parser.add_argument('--wandb_project', type=str, default='sam-demo')
    parser.add_argument('--experiment_name', type=str, default=None)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)
