# Reproduce PIPs Results

## Environment Setup

I will use conda:

```bash
conda create --name pips python=3.10 -y
source activate pips
conda install pytorch=1.12.0 torchvision=0.13.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

## Run Provided Demo

Download the reference model and run the demo. This will also make sure
that everything was set up correctly. For example:

```bash
bash get_reference_model.sh
python chain_demo.py
```

## Run on Your Own Video

Prepare a video of your own. For example, I will process a few videos such as [this youtube
video](https://www.youtube.com/watch?v=gqHy_trMnRk&ab_channel=cro3x3) that I have downloaded 
and put into `demo_images/ulaz_u_crnu_riku.mp4`:

```bash
video_to_images() {
  my_video_path=$1
  output_images_path=$2
  output_fps=$3
  if [ -f "$my_video_path" ]; then
    mkdir -p ${output_images_path}
    ffmpeg -loglevel panic -i "${my_video_path}" -q:v 1 -vf fps=$output_fps "${output_images_path}"/%06d.jpg
  else
    echo "$my_video_path does not exist."
  fi
}

video_to_images "./demo_images/ulaz_u_crnu_riku.mp4" "./demo_images/ulaz_u_crnu_riku" 24
video_to_images "./demo_images/frano.avi" "./demo_images/frano" 24
video_to_images "./demo_images/slow.flv" "./demo_images/slow" 24
```

I will select a subset of images to run with the chain_demo:

```bash
select_images() {
  output_images_path=$1
  selected_images_output_path=$2
  images_from=$3
  images_to=$4
  if [ -f "$my_video_path" ]; then
    mkdir -p ${selected_images_output_path}
    for idx in $(seq ${images_from} ${images_to}); do
      filename=$(printf "%06d" ${idx}).jpg
      echo ${output_images_path}/${filename} "-->" ${selected_images_output_path}/${filename}
      \cp ${output_images_path}/${filename} ${selected_images_output_path}/${filename}
    done
  fi
}
select_images "./demo_images/ulaz_u_crnu_riku" "./demo_images/ulaz_u_crnu_riku__selected" 504 630
```

Finally, run on your own video:

```bash
python demo.py \
  --input_images_path "./demo_images/ulaz_u_crnu_riku" \
  --output_gifs_path "./demo_output/ulaz_u_crnu_riku"
python demo.py \
  --input_images_path "./demo_images/ulaz_u_crnu_riku__selected" \
  --output_gifs_path "./demo_output/ulaz_u_crnu_riku__selected"
python demo.py \
  --input_images_path "./demo_images/frano" \
  --output_gifs_path "./demo_output/frano"
python demo.py \
  --input_images_path "./demo_images/slow" \
  --output_gifs_path "./demo_output/slow"

python chain_demo.py \
  --input_images_path "./demo_images/ulaz_u_crnu_riku__selected" \
  --output_gifs_path "./demo_output/ulaz_u_crnu_riku__selected"
python chain_demo.py \
  --input_images_path "./demo_images/frano" \
  --output_gifs_path "./demo_output/frano"
```

## Prepare Data

I will download the necessary data into `./data`, which I soft linked to
a data folder using `ln -s /scratch/izar/rajic/eth-master-thesis/00-data
data`.

### 0. Set up a torrent client

To download the [FlyingThings
dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html),
we need to have a torrent client. I will install `qBittorrent-nox`, a
version of qBittorrent (Qt5 application) that does not require X and can
be controlled via a WebUI. I will fetch a static build of
`qBittorrent-nox` from
[qbittorrent-nox-static](https://github.com/userdocs/qbittorrent-nox-static).

```bash
# Download the binary
mkdir -p ~/bin && source ~/.profile
wget -qO ~/bin/qbittorrent-nox https://github.com/userdocs/qbittorrent-nox-static/releases/latest/download/x86_64-qbittorrent-nox
chmod 700 ~/bin/qbittorrent-nox

# Configure the client
mkdir -p ~/.config/qBittorrent
cat > ~/.config/qBittorrent/qBittorrent.conf <<EOL
[LegalNotice]
Accepted=true

[Preferences]
General\Locale=en_GB
WebUI\Port=10722
WebUI\HostHeaderValidation=false
Downloads\SavePath=private/qBittorrent/data
EOL

cat ~/.config/qBittorrent/qBittorrent.conf

# Launch the web client
qbittorrent-nox
# WebUI will be started shortly after internal preparations. Please wait...
# ******** Information ********
# To control qBittorrent, access the WebUI at: http://localhost:10722
# ...
```

### 1. FlyingThings++

Download the trajectory and occlusion data that was used in the paper
(or produce it on your own following the official `README.md`):

```bash
mkdir -p data/flyingthings
cd data/flyingthings

pip install gdown
gdown --no-check-certificate --folder https://drive.google.com/drive/folders/1zzWkGGFgJPyHpVaSA19zpYlux1Mf6wGC
tar xvfz occluders_al.tar.gz
cat trajs_ad.tar.gz.* | tar xvfz -
rm -rf occluders_al.tar.gz
rm -rf trajs_ad.tar.gz.a*

cd -
```

Additionally, download the `RGB images (cleanpass) (WebP)` portion of
the [FlyingThings
dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html),
using a torrent client. Also, download the `Object segmentation` data
using wget. In case you will produce the trajectory and occlusion data
yourself, then you also need to download the `Optical flow` data.

The structure should look like this:

```bash
tree -L 3 data
# data
# └── flyingthings
#     ├── frames_cleanpass_webp
#     │   ├── TEST
#     │   └── TRAIN
#     ├── object_index
#     │   ├── TEST
#     │   └── TRAIN
#     ├── occluders_al
#     │   ├── TEST
#     │   └── TRAIN
#     └── trajs_ad
#         ├── TEST
#         └── TRAIN

du -sch data/flyingthings # To get a feeling about the size
# 7.5G    data/flyingthings/frames_cleanpass_webp
# 104G    data/flyingthings/object_index
# 6.8G    data/flyingthings/occluders_al
# 78G     data/flyingthings/trajs_ad
# 196G    total
```

_Nota bene_: the suffixes "ad" and "al" are dataset (pre)processing
version counters from the authors (aa, ab, ac, ...).

### 2. CroHD

Follow [`README.md`](https://github.com/m43/pips#crohd) instructions:

```bash
cd data

wget https://motchallenge.net/data/HT21.zip
unzip HT21.zip
rm HT21.zip

wget https://motchallenge.net/data/HT21Labels.zip
unzip HT21Labels.zip
rm HT21Labels.zip

mkdir head_tracking
mv HT21 head_tracking/HT21
mv HT21Labels head_tracking/HT21Labels
```

### 3. DAVIS

Follow [`README.md`](https://github.com/m43/pips#davis) instructions:

```bash
cd data

filenames=(
  DAVIS-2017-trainval-Full-Resolution.zip
  DAVIS-2017-test-dev-Full-Resolution.zip
  DAVIS-2017-test-challenge-Full-Resolution.zip
  DAVIS-2017-trainval-480p.zip
  DAVIS-2017-test-dev-480p.zip
  DAVIS-2017-test-challenge-480p.zip
)

for filename in ${filenames[@]}; do
  wget https://data.vision.ee.ethz.ch/csergi/share/davis/$filename --no-check-certificate
  unzip -n $filename
  rm $filename
done
```

### 4. BADJA

Follow [`README.md`](https://github.com/m43/pips#badja) instructions:

```bash
DAVIS_PATH=/scratch/izar/rajic/eth-master-thesis/00-data/DAVIS

cd data
git clone https://github.com/benjiebob/BADJA.git badja_data
cd badja_data

if [ -d "${DAVIS_PATH}" ]; then
  ln -s "${DAVIS_PATH}" DAVIS
  gdown https://drive.google.com/uc?id=1ad1BLmzyOp_g3BfpE2yklNI-E1b8y4gy
  unzip badja_extra_videos.zip
  rm badja_extra_videos.zip
else
  echo "Download DAVIS first to ${DAVIS_PATH}"
fi
```

### 5. Verify Data Setup

The final data structure would look as follows:

```bash
tree -L 3 data
# data
# ├── badja_data
# │   ├── code
# │   │   ├── badja_data.py
# │   │   ├── joint_catalog.py
# │   │   └── view_badja.py
# │   ├── DAVIS -> /scratch/izar/rajic/eth-master-thesis/00-data/DAVIS
# │   ├── extra_videos
# │   │   ├── impala0
# │   │   └── rs_dog
# │   ├── gifs
# │   │   ├── bear.gif
# │   │   ├── camel.gif
# │   │   ├── cows.gif
# │   │   ├── dog.gif
# │   │   ├── horsejump-high.gif
# │   │   ├── horsejump-low.gif
# │   │   ├── impala0.gif
# │   │   └── rs_dog.gif
# │   ├── joint_annotations
# │   │   ├── bear.json
# │   │   ├── camel.json
# │   │   ├── cat_jump.json
# │   │   ├── cows.json
# │   │   ├── dog-agility.json
# │   │   ├── dog.json
# │   │   ├── horsejump-high.json
# │   │   ├── horsejump-low.json
# │   │   ├── impala0.json
# │   │   ├── rs_dog.json
# │   │   └── tiger.json
# │   ├── LICENSE
# │   └── README.md
# ├── DAVIS
# │   ├── Annotations
# │   │   ├── 480p
# │   │   └── Full-Resolution
# │   ├── ImageSets
# │   │   ├── 2016
# │   │   └── 2017
# │   ├── JPEGImages
# │   │   ├── 480p
# │   │   └── Full-Resolution
# │   ├── README.md
# │   └── SOURCES.md
# ├── flyingthings
# │   ├── frames_cleanpass_webp
# │   │   ├── TEST
# │   │   └── TRAIN
# │   ├── object_index
# │   │   ├── TEST
# │   │   └── TRAIN
# │   ├── occluders_al
# │   │   ├── TEST
# │   │   └── TRAIN
# │   └── trajs_ad
# │       ├── TEST
# │       └── TRAIN
# └── head_tracking
#     ├── HT21
#     │   ├── test
#     │   └── train
#     └── HT21Labels
#         ├── test
#         └── train

du -sch data/flyingthings data/head_tracking data/DAVIS data/badja_data
# 196G    data/flyingthings
# 4.5G    data/head_tracking
# 7.1G    data/DAVIS
# 554M    data/badja_data
```

To make sure the data is downloaded and placed correctly, you can run
the following evaluation commands, one for each dataset. The evaluation
script will use the checkpoint saved in `reference_model` by default
(e.g., the `reference_model/model-000200000.pth`).

```bash
cd path/to/pips/project

python test_on_flt.py
python test_on_crohd.py
python test_on_davis.py
python test_on_badja.py
```

## Train

I will train on 2 GPUs, a batch size of 2 instead of 4, and 128
points/trajectories instead of 768:

```bash
python train.py --horz_flip=True --vert_flip=True --device_ids=[0,1] --B=2 --N=128
```

If training crashes, you can continue for example as:
```bash
python train.py --horz_flip=True --vert_flip=True --device_ids=[0,1] --B=2 --N=128 \
  --init_dir "logs/my_pips/checkpoints/8hv_8_128_I4_5e-4_A_debug_15:22:55" \
  --load_optimizer True \
  --load_step True
```

## Reproduce Paper Numbers

### Evaluating the Provided Checkpoint

Using the provided checkpoint should give numbers close to those
provided in the `README.md`. However, the numbers should not match those
in the paper, since the checkpoint was improved since the paper's
publication (by using a harder version of FlyingThings++ to train on).
To reproduce the numbers run the following:

```bash
python test_on_flt.py
# model_name 1_8_16_pips_flt_16:05:08
# loading FlyingThingsDataset...found 2542 samples in data/flyingthings (d
# set=TEST, subset=all, version=ad)
# loading occluders...found 1631 occluders in data/flyingthings (dset=TEST, subset=all, version=al)
# reading ckpt from reference_model
# ...found checkpoint reference_model/model-000200000.pth
# setting max_iters 2542
# ...
# 1_8_16_pips_flt_16:05:08; step 002539/2542; rtime 0.00; itime 0.11, ate_vis = 6.08, ate_occ = 19.36
# 1_8_16_pips_flt_16:05:08; step 002540/2542; rtime 0.00; itime 0.11, ate_vis = 6.08, ate_occ = 19.36
# 1_8_16_pips_flt_16:05:08; step 002541/2542; rtime 0.00; itime 0.11, ate_vis = 6.07, ate_occ = 19.36
# 1_8_16_pips_flt_16:05:08; step 002542/2542; rtime 0.00; itime 0.11, ate_vis = 6.08, ate_occ = 19.35
# TODO: re-evaluate

python test_on_crohd.py
# 1_8_128_pips_vis_crohd_20:27:19 step=000237/237 readtime=0.06 itertime=0.38 ate_all=4.86 ate_vis=4.82 ate_occ=11.79

python test_on_badja.py
# model_name 1_8_pips_badja_15:09:30
# annotations_path data/badja_data/joint_annotations
# number of annotated frames 17
# number of annotated frames 18
# number of annotated frames 18
# number of annotated frames 21
# number of annotated frames 5
# number of annotated frames 12
# number of annotated frames 10
# number of annotated frames 12
# number of annotated frames 29
# number of annotated frames 201
# number of annotated frames 10
# Loaded BADJA dataset
# found 7 unique videos in data/badja_data
# reading ckpt from ./reference_model
# ...found checkpoint ./reference_model/model-000200000.pth
#
# 1_8_pips_badja_15:09:30; step 000001/7; rtime 5.97; itime 23.99; bear; pck 76.4
# results ['76.4', 'avg 76.4']
# 1_8_pips_badja_15:09:30; step 000002/7; rtime 0.11; itime 27.73; camel; pck 91.6
# results ['76.4', '91.6', 'avg 84.0']
# 1_8_pips_badja_15:09:30; step 000003/7; rtime 0.09; itime 30.16; cows; pck 87.2
# results ['76.4', '91.6', '87.2', 'avg 85.1']
# 1_8_pips_badja_15:09:30; step 000004/7; rtime 0.13; itime 13.80; dog-agility; pck 31.0
# results ['76.4', '91.6', '87.2', '31.0', 'avg 71.6']
# 1_8_pips_badja_15:09:30; step 000005/7; rtime 0.02; itime 18.79; dog; pck 46.0
# results ['76.4', '91.6', '87.2', '31.0', '46.0', 'avg 66.4']
# 1_8_pips_badja_15:09:30; step 000006/7; rtime 0.06; itime 18.80; horsejump-high; pck 62.3
# results ['76.4', '91.6', '87.2', '31.0', '46.0', '62.3', 'avg 65.7']
# 1_8_pips_badja_15:09:30; step 000007/7; rtime 0.05; itime 20.42; horsejump-low; pck 61.3
# results ['76.4', '91.6', '87.2', '31.0', '46.0', '62.3', '61.3', 'avg 65.1']
# TODO: re-evaluate
```

### Evaluating the Trained Model

To evaluate the model we have trained, find the location of the relevant
checkpoint and run:

```bash
`TODO: finish training`
```

### Evaluating Baselines

Evaluating the baselines will exactly reproduce the numbers reported in
the paper for BADJA (table 4), but not for CroHD or FlyingThings++.

Evaluate DINO, the checkpoint will be downloaded automatically:
```bash
python test_on_flt.py   --modeltype='dino'
# TODO: re-evaluate

python test_on_crohd.py --modeltype='dino'
# TODO: re-evaluate

python test_on_badja.py --modeltype='dino'
# 1_8_dino_badja_23:05:16; step 000001/7; rtime 7.42; itime 8.50; bear; pck 75.0
# 1_8_dino_badja_23:05:16; step 000002/7; rtime 2.34; itime 7.02; camel; pck 59.2
# 1_8_dino_badja_23:05:16; step 000003/7; rtime 6.04; itime 8.07; cows; pck 70.6
# 1_8_dino_badja_23:05:16; step 000004/7; rtime 0.11; itime 1.58; dog-agility; pck 10.3
# 1_8_dino_badja_23:05:16; step 000005/7; rtime 0.03; itime 4.28; dog; pck 47.1
# 1_8_dino_badja_23:05:16; step 000006/7; rtime 1.01; itime 4.07; horsejump-high; pck 35.1
# 1_8_dino_badja_23:05:16; step 000007/7; rtime 3.30; itime 4.72; horsejump-low; pck 56.0
# results ['75.0', '59.2', '70.6', '10.3', '47.1', '35.1', '56.0', 'avg 50.5']
# TODO: re-evaluate
```

Evaluate RAFT, but first download a checkpoint from their
[GitHub](https://github.com/princeton-vl/RAFT):
```bash
wget https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip
unzip models.zip
rm models.zip
mv models raft_ckpts
lss raft_ckpts
# total 85M
# -rw-r--r--   21M   Jul 25 2020   raft-chairs.pth
# -rw-r--r--   21M   Jul 25 2020   raft-kitti.pth
# -rw-r--r--   21M   Jul 25 2020   raft-sintel.pth
# -rw-rw-r--  3.9M   Aug 24 2020   raft-small.pth
# -rw-r--r--   21M   Jul 25 2020   raft-things.pth

python test_on_flt.py   --modeltype='raft'
# TODO: re-evaluate

python test_on_crohd.py --modeltype='raft'
# 1_8_128_raft_vis_crohd_20:27:19 step=000237/237 readtime=0.05 itertime=1.60 ate_all=8.23 ate_vis=8.20 ate_occ=15.24

python test_on_badja.py --modeltype='raft'
# 1_8_raft_badja_23:29:59; step 000001/7; rtime 5.83; itime 16.02; bear; pck 64.6
# 1_8_raft_badja_23:29:59; step 000002/7; rtime 0.09; itime 12.92; camel; pck 65.6
# 1_8_raft_badja_23:29:59; step 000003/7; rtime 0.09; itime 17.82; cows; pck 69.5
# 1_8_raft_badja_23:29:59; step 000004/7; rtime 0.11; itime 3.01; dog-agility; pck 13.8
# 1_8_raft_badja_23:29:59; step 000005/7; rtime 0.02; itime 7.85; dog; pck 39.1
# 1_8_raft_badja_23:29:59; step 000006/7; rtime 0.06; itime 10.19; horsejump-high; pck 37.1
# 1_8_raft_badja_23:29:59; step 000007/7; rtime 0.07; itime 10.16; horsejump-low; pck 29.3
# results ['64.6', '65.6', '69.5', '13.8', '39.1', '37.1', '29.3', 'avg 45.6']
# TODO: re-evaluate
```

## Plot Precision Plots

I will create more verbose summaries and figures to better inspect the
performance the different methods and better compare them.

### FlyingThings++

For this, I first create and save evaluation summaries to disk:
```bash
python test_on_flt.py --modeltype='pips'
# logs_test_on_flt/1_8_16_pips_flt_10:48:30/results_df.csv

python test_on_flt.py --modeltype='raft'
# logs_test_on_flt/1_8_16_raft_flt_02:04:56/results_df.csv

python test_on_flt.py --modeltype='dino'
# logs_test_on_flt/1_8_16_dino_flt_02:03:12/results_df.csv
```

With the `csv` results dataframes ready, I will now run the creation of
the figures and other summaries:
```bash
python -m pips_utils.figures \
  --mostly_visible_threshold 4 \
  --results_df_path_list \
  logs_test_on_flt/1_8_16_pips_flt_10:48:30/results_df.csv \
  logs_test_on_flt/1_8_16_raft_flt_02:04:56/results_df.csv \
  logs_test_on_flt/1_8_16_dino_flt_02:03:12/results_df.csv \
  --results_name_list \
  PIPs \
  RAFT \
  DINO
```

### CroHD

For CroHD, we evaluate on the same metrics and create the same plots as
for FlytingThings++:
```bash
python test_on_crohd.py --modeltype='pips'
python test_on_crohd.py --modeltype='raft'
python test_on_crohd.py --modeltype='dino'

python -m pips_utils.figures \
  --mostly_visible_threshold 8 \
  --results_df_path_list \
  logs_test_on_crohd/1_8_128_pips_vis_crohd_20:27:19/results_df.csv \
  logs_test_on_crohd/1_8_128_raft_vis_crohd_20:27:19/results_df.csv \
  --results_name_list \
  PIPs \
  RAFT
```

## Plot Dataset Annotations

`TODO`

## Identifying Failure Cases

`TODO`
