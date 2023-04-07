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

## SAM + Predicted Trajectories

Run SAM:

```bash
for x in experiments/group_a/experiment_01/sbatch_files/A1_00*.sh; do sbatch $x; done

# python -m evaluate --modeltype pips --dataset_type tapvid --dataset_location data/tapvid_davis/tapvid_davis.pkl --subset davis --query_mode first --log_freq 1 --max_iter 3 --wandb_project evaluation-tapvid-trajectories-with-sam --log_sam_output
# python -m evaluate --modeltype pips --dataset_type tapvid --dataset_location none --subset kubric --query_mode first --log_freq 1 --max_iter 3 --wandb_project evaluation-tapvid-trajectories-with-sam --log_sam_output
# python -m evaluate --modeltype pips --dataset_type tapvid --dataset_location data/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl --subset rgb_stacking --query_mode first --log_freq 1 --max_iter 3 --wandb_project evaluation-tapvid-trajectories-with-sam --log_sam_output
# python -m evaluate --modeltype pips --dataset_type tapvid --dataset_location data/tapvid_kinetics --subset kinetics --query_mode first --log_freq 1 --max_iter 3 --wandb_project evaluation-tapvid-trajectories-with-sam --log_sam_output

# python -m evaluate --modeltype raft --dataset_type tapvid --dataset_location data/tapvid_davis/tapvid_davis.pkl --subset davis --query_mode first --log_freq 1 --max_iter 3 --wandb_project evaluation-tapvid-trajectories-with-sam --log_sam_output
# python -m evaluate --modeltype raft --dataset_type tapvid --dataset_location none --subset kubric --query_mode first --log_freq 1 --max_iter 3 --wandb_project evaluation-tapvid-trajectories-with-sam --log_sam_output
# python -m evaluate --modeltype raft --dataset_type tapvid --dataset_location data/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl --subset rgb_stacking --query_mode first --log_freq 1 --max_iter 3 --wandb_project evaluation-tapvid-trajectories-with-sam --log_sam_output
# python -m evaluate --modeltype raft --dataset_type tapvid --dataset_location data/tapvid_kinetics --subset kinetics --query_mode first --log_freq 1 --max_iter 3 --wandb_project evaluation-tapvid-trajectories-with-sam --log_sam_output
```

## Set up Latex

Followed instructions from the `tl;dr: Unix(ish)` section [here](https://www.tug.org/texlive/quickinstall.html).

```bash
cd /scratch/frrajic/tmp # working directory of your choice
wget https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz # or curl instead of wget
zcat < install-tl-unx.tar.gz | tar xf -
rm install-tl-*.tar.gz
cd install-tl-*
perl ./install-tl --no-interaction --texdir=/scratch_net/biwidl217/frrajic/texlive --texuserdir=/scratch_net/biwidl217/frrajic/texlive_userdir

# Prepend /path/to/texlive/YYYY/bin/PLATFORM to your PATH,
echo 'export PATH="/scratch_net/biwidl217/frrajic/texlive/bin/x86_64-linux:$PATH"' >> ~/.bashrc
```
