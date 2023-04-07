# Using Point-Tracking Methods Together With Sam

## Setup SAM

Set up the environment:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install pycocotools onnxruntime onnx
```

Download three different checkpoints:

```bash
mkdir sam_checkpoints
cd sam_checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
md5sum sam_vit_h_4b8939.pth
md5sum sam_vit_l_0b3195.pth
md5sum sam_vit_b_01ec64.pth
cd -
```

## SAM Demo

Run on demo images to make sure it is set up correctly:

```bash
python -m demo_sam --input_video_folder demo_images/black_french_bulldog
python -m demo_sam --input_video_folder demo_images/frano
python -m demo_sam --input_video_folder demo_images/ulaz_u_crnu_riku__selected
```

