import argparse
import os
import time

import cv2
import imageio
import numpy as np
import skimage.io
from pycocotools.ytvos import YTVOS
from tqdm import tqdm

from pips_utils.util import setup_wandb, log_video_to_wandb, ensure_dir


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--video_idx', type=int, default=None)
    parser.add_argument('--video_frames_dir', type=str, default='./data/UVOv1.0/uvo_videos_dense_frames/')
    parser.add_argument('--annotations_json', type=str,
                        default='./data/UVOv1.0/VideoDenseSet/UVO_video_val_dense.json')
    parser.add_argument('--predictions_json', type=str,
                        default='./data/UVOv1.0/ExampleSubmission/video_val_pred.json')

    # Logging
    parser.add_argument('--wandb_entity', type=str, default='point-tracking')
    parser.add_argument('--wandb_project', type=str, default='visualize-uvo')
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--local_gif_output_path', type=str, default=None)
    parser.add_argument('--fps', type=int, default=30)

    return parser


def main(args):
    setup_wandb(args.wandb_entity, args.wandb_project, args.experiment_name)
    uvo_api, uvo_det = load_uvo(args.annotations_json, args.predictions_json)
    all_video_ids = uvo_api.getVidIds()
    if args.video_idx is None:
        video_indices = all_video_ids
    else:
        assert args.video_idx in all_video_ids, f"video_idx {args.video_idx} not in {all_video_ids}"
        video_indices = [args.video_idx]
    for video_idx in tqdm(video_indices):
        visualize_uvo_video(uvo_api, uvo_det, video_idx, args.video_frames_dir,
                            args.local_gif_output_path, args.fps)


def visualize_uvo_video(uvo_api, uvo_det, video_idx, video_frames_dir, local_logs_dir, fps):
    # Input
    print(f"Loading frames for video {video_idx}")
    tic = time.time()
    frames = load_input_video_frames(uvo_api, video_frames_dir, video_idx)
    print(f"Done (t={time.time() - tic:0.2f}s)\n")
    # GT
    print(f"Loading GT masks for video {video_idx}")
    tic = time.time()
    gt_masked_frames = mask_video_frames_with_annotations(uvo_api, video_frames_dir, video_idx, frames)
    print(f"Done (t={time.time() - tic:0.2f}s)\n")
    # Pred
    print(f"Loading Pred masks for video {video_idx}")
    tic = time.time()
    pred_masked_frames = mask_video_frames_with_annotations(uvo_det, video_frames_dir, video_idx, frames)
    print(f"Done (t={time.time() - tic:0.2f}s)\n")
    # [Input, GT, Pred]
    concat_masked_frames = [
        np.concatenate((frame, gt_masked_frame, pred_masked_frame), axis=0)
        for frame, gt_masked_frame, pred_masked_frame
        in zip(frames, gt_masked_frames, pred_masked_frames, strict=True)
    ]
    # Save as GIFs
    print(f"Saving GIFs for video {video_idx}")
    tic = time.time()
    log_video_to_wandb(f"input", frames=frames, fps=fps, step=video_idx)
    log_video_to_wandb(f"gt_masks", frames=gt_masked_frames, fps=fps, step=video_idx)
    log_video_to_wandb(f"pred_masks", frames=pred_masked_frames, fps=fps, step=video_idx)
    log_video_to_wandb(f"all_masks", frames=concat_masked_frames, fps=fps, step=video_idx)
    if local_logs_dir is not None:
        ensure_dir(local_logs_dir)
        imageio.mimsave(os.path.join(local_logs_dir, f"{video_idx}__input.gif"), frames, fps=fps)
        imageio.mimsave(os.path.join(local_logs_dir, f"{video_idx}__gt_masks.gif"), gt_masked_frames, fps=fps)
        imageio.mimsave(os.path.join(local_logs_dir, f"{video_idx}__pred_masks.gif"), pred_masked_frames, fps=fps)
        imageio.mimsave(os.path.join(local_logs_dir, f"{video_idx}__all_masks.gif"), concat_masked_frames, fps=fps)
    print(f"Done (t={time.time() - tic:0.2f}s)\n")


def load_uvo(annotations_json, predictions_json):
    uvo_api: YTVOS = YTVOS(annotations_json)
    uvo_det: YTVOS = uvo_api.loadRes(predictions_json)

    # convert ann in uvo_det to class-agnostic
    for ann in uvo_det.dataset["annotations"]:
        if ann["category_id"] != 1:
            ann["category_id"] = 1

    return uvo_api, uvo_det


def load_input_video_frames(uvo_api, video_frames_dir, video_idx):
    vid = uvo_api.loadVids(ids=[video_idx])[0]
    frames = [
        skimage.io.imread(os.path.join(video_frames_dir, frame_path))
        for frame_path in vid["file_names"]
    ]
    return frames


def get_random_segmentation_colors(n):
    return (np.random.random((n, 3)) * 0.6 + 0.4).tolist()


def mask_video_frames_with_annotations(
        uvo_api: YTVOS,
        video_frames_dir: str,
        video_idx: int,
        frames=None,
        mask_color_fn=get_random_segmentation_colors,
):
    if frames is None:
        frames = load_input_video_frames(uvo_api, video_frames_dir, video_idx)

    ann_ids = uvo_api.getAnnIds(vidIds=[video_idx])
    anns = uvo_api.loadAnns(ids=ann_ids)
    mask_colors = mask_color_fn(len(ann_ids))
    masks = annotations_to_masks(uvo_api, anns)
    masked_frames = add_masks_to_frames(frames, masks, mask_colors)
    return masked_frames


def annotations_to_masks(uvo_api, annotations):
    masks = [
        [
            uvo_api.annToMask(ann, frame) if ann["areas"][frame] is not None else None
            for frame in range(len(ann["areas"]))
        ]
        for ann in annotations
    ]
    return masks


def add_mask_to_frame(frame, mask, color):
    if mask is None:
        return frame

    frame = frame / 255

    overlay = np.ones((mask.shape[0], mask.shape[1], 3))
    overlay *= color
    overlay = np.where(mask[..., None] > 0, overlay, frame)

    masked_frame = cv2.addWeighted(frame, 0.2, overlay, 0.8, 0)
    masked_frame = masked_frame * 255
    masked_frame = masked_frame.astype(np.uint8)
    return masked_frame


def add_masks_to_frame(frame, masks, colors):
    for mask, color in zip(masks, colors):
        frame = add_mask_to_frame(frame, mask, color)
    return frame


def add_masks_to_frames(frames, masks, colors):
    masked_frames = []
    for i, frame in enumerate(frames):
        frame_masks = [mask[i] for mask in masks]
        masked_frames.append(add_masks_to_frame(frame, frame_masks, colors))
    return masked_frames


if __name__ == '__main__':
    main(get_parser().parse_args())
