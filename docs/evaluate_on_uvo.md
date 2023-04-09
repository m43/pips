# Evaluation on the Object Tracking Task of the UVO Dataset	

This markdown file documents how to prepare the UVO dataset and evaluate
PIPS+Sam and RAFT+Sam on it.

## Prepare UVO

First, go to your data root directory, for example `cd ./data/`. We will
download the dataset into a subfolder named `UVOv1.0`.

Download the preprocessed UVO dataset. Note that I have zipped and
reuploaded the UVOv1 folder for a more convenient download from Google
Drive as it is otherwise hard to download a folder with a lot of files
from Google Drive using gdown. I have downloaded and zipped the folder
on April 8, 2023.

```bash
# Download the Annotations and other utilities
pip install gdown
gdown --no-check-certificate https://drive.google.com/uc?id=1AGu4BL-i_vDCMNtwsoSuo5wIyDVd5dRf
unzip  UVOv1.0.zip
rm UVOv1.0.zip

# Download the preprocessed videos
gdown --no-check-certificate --folder https://drive.google.com/drive/folders/1fOhEdHqrp_6D_tBsrR9hazDLYV2Sw1XC
unzip UVO_Videos/uvo_videos_dense.zip
unzip UVO_Videos/uvo_videos_sparse.zip
mv uvo_videos_dense/ UVOv1.0/
mv uvo_videos_sparse/ UVOv1.0/
rm -rf UVO_Videos/
rm -rf __MACOSX/

tree -L 1 UVOv1.0
# UVOv1.0
# ├── EvaluationAPI
# ├── ExampleSubmission
# ├── FrameSet
# ├── README.md
# ├── VideoDenseSet
# ├── VideoSparseSet
# ├── YT-IDs
# ├── download_kinetics.py
# ├── preprocess_kinetics.py
# ├── uvo_videos_dense
# ├── uvo_videos_sparse
# └── video2frames.py

du -sch UVOv1.0/*
# 6.3M    UVOv1.0/EvaluationAPI
# 683M    UVOv1.0/ExampleSubmission
# 176M    UVOv1.0/FrameSet
# 8.0K    UVOv1.0/README.md
# 736M    UVOv1.0/VideoDenseSet
# 3.5G    UVOv1.0/VideoSparseSet
# 248K    UVOv1.0/YT-IDs
# 4.0K    UVOv1.0/download_kinetics.py
# 8.0K    UVOv1.0/preprocess_kinetics.py
# 1.3G    UVOv1.0/uvo_videos_dense
# 13G     UVOv1.0/uvo_videos_sparse
# 4.0K    UVOv1.0/video2frames.py
# 20G     total
```

I recommend replacing the `UVOv1.0/video2frames.py` script with:
```py
import argparse
import cv2
import os
import pathlib
from tqdm import tqdm

def split_single_video(video_path, frames_dir=""):
    cap = cv2.VideoCapture(video_path)
    cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            success, buffer = cv2.imencode(".png", frame)
            if success:
                with open(f"{frames_dir}{cnt}.png", "wb") as f:
                    f.write(buffer.tobytes())
                    f.flush()
                cnt += 1
        else:
            break
    return cnt


def get_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--video_dir", type=str, default="NonPublic/uvo_videos_dense/")
    arg_parser.add_argument("--frames_dir", type=str, default="NonPublic/uvo_videos_dense_frames/")
    return arg_parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    video_paths = os.listdir(args.video_dir)
    print(f"Splitting videos in {args.video_dir} to frames in {args.frames_dir}...")
    print(f"Total number of videos: {len(video_paths)}")
    for video_path in tqdm(video_paths):
        print(f"Splitting {video_path}...")
        v_frame_dir = pathlib.Path(os.path.join(args.frames_dir, video_path[:-4]))
        if not v_frame_dir.is_dir():
            v_frame_dir.mkdir(parents=True, exist_ok=False)
        n_frames = split_single_video(os.path.join(args.video_dir, video_path), frames_dir=v_frame_dir)
        print(f"Total number of frames extracted from {video_path}: {n_frames}")
    print(f"Done.")

```

Then you can split the preprocessed videos into frames by running:

```bash
python UVOv1.0/video2frames.py --video_dir UVOv1.0/uvo_videos_dense --frames_dir UVOv1.0/uvo_videos_dense_frames
python UVOv1.0/video2frames.py --video_dir UVOv1.0/uvo_videos_sparse --frames_dir UVOv1.0/uvo_videos_sparse_frames
```

Install the VideoAPI Python package from `youtubevos`. Note that this
package is also added as a requirement in `requirements.txt`.

```bash
pip install cython
pip uninstall pycocotools # Remove previous package, if there is any
pip install git+https://git@github.com/youtubevos/cocoapi.git@f24b5f58594adfe4f4c015bf49dbc819cc3be98f#subdirectory=PythonAPI

## Alternatively:
# cd EvaluationAPI/VideoAPI/cocoapi/PythonAPI
# pip install cython
# python setup.py build_ext install
# cd -
```

You can read more details about the dataset in the `README.md` that the
authors provided, or on their [website](https://sites.google.com/view/unidentified-video-object/dataset). There are additional
`README.md` files in subdirectories of the dataset.

Finally, go back to the pips root directory with `cd /path/to/root/of/pips/`.

## Sam Demo on a UVO Video

Running Sam to automatically segment everything in the a `854x480`
beekeeper video from UVO. Note that the UVO videos are actually videos
from the Kinetics dataset which were fetched from YouTube, and the
beekeeper video can for example be found directly [here](https://www.youtube.com/watch?v=-18X6h92xpw) on YouTube.

```bash
python -m demo_sam --input_video_folder data/UVOv1.0/uvo_videos_dense_frames/-18X6h92xpw
```

