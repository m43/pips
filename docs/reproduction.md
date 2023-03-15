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

Prepare a video of your own. For example, I will process the video I
have downloaded
from [this youtube video](https://www.youtube.com/watch?v=gqHy_trMnRk&ab_channel=cro3x3)
and put it into `demo_images/ulaz_u_crnu_riku.mp4`:

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

I will select a smaller number of images to run with the chain_demo:

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

Run on your own video:

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

I will download the necessary data into `./data`, that I soft linked
to a data folder
using `ln -s /scratch/izar/rajic/eth-master-thesis/00-data data`.

Download the exact FlyingThings++ dataset as used in the paper (or
produce it on your own following the official `README.md`):

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

The structure should look like this:

```bash
tree -L 2 data
#data
#└── flyingthings
#    ├── occluders_al
#    │   ├── TEST
#    │   │   ├── A
#    │   │   ├── B
#    │   │   └── C
#    │   └── TRAIN
#    │       ├── A
#    │       ├── B
#    │       └── C
#    └── trajs_ad
#        ├── TEST
#        │   ├── A
#        │   ├── B
#        │   └── C
#        └── TRAIN
#            ├── A
#            ├── B
#            └── C
```

_Nota bene_: the suffixes "ad" and "al" are dataset (pre)processing
version counters from the authors (aa, ab, ac, ...).

## Train

To reproduce the result in the paper, you should train with 4 gpus,
with horizontal and vertical flips, with a command like this:

```bash
#python train.py --horz_flip=True --vert_flip=True --device_ids=[0,1]
python train.py --horz_flip=True --vert_flip=True --device_ids=[0,1,2,3]
```

## Reproduce Paper Numbers

`TODO`

```bash
python test_on_flt.py
```

## Plot Precision Plots

`TODO`

## Plot Dataset Annotations

`TODO`

## Identifying Failure Cases

`TODO`
