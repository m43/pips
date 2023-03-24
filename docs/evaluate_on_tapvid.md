# Evaluate on the TAP-Vid Dataset

## Prepare the dataset

I follow the instructions given in
[deepmind/tapnet](https://github.com/deepmind/tapnet/tree/main/data).
This is the sequence of commands I had run to set up the dataset
correctly:
```bash
cd data
for zipfile in tapvid_davis.zip tapvid_rgb_stacking.zip tapvid_kinetics.zip; do
  wget https://storage.googleapis.com/dm-tapnet/${zipfile}
  unzip $zipfile
  rm $zipfile
done

# I took Kinetics-700-2020 download links from https://github.com/cvdfoundation/kinetics-dataset
curr_dl="k700-2020_targz/val"
curr_extract="k700-2020/val"
mkdir -p $curr_dl
mkdir -p $curr_extract
wget -c -i https://s3.amazonaws.com/kinetics/700_2020/val/k700_2020_val_path.txt -P $curr_dl
for f in $(ls $curr_dl); do
  [[ $f == *.tar.gz ]] && echo Extracting $curr_dl/$f to $curr_extract && tar zxf $curr_dl/$f -C $curr_extract
done

# Process the downloaded Kinetics-700-2020 videos into a pickle file
wget https://raw.githubusercontent.com/deepmind/tapnet/b8e0641e3c6a3483060e49df1def87fef16c8d1a/data/generate_tapvid.py
pip install ffmpeg-python
python3 generate_tapvid.py \
  --input_csv_path=tapvid_kinetics/tapvid_kinetics.csv \
  --output_base_path=tapvid_kinetics/ \
  --video_root_path=k700-2020/val \
  --alsologtostderr

# Delete the .tar.gz files, if you want to
# rm -rf $curr_extract

# You can also delete the raw kinetics videos if you want
# rm -rf $curr_dl

tree -L 2
# .
# ├── tapvid_davis
# │   ├── README.md
# │   ├── SOURCES.md
# │   └── tapvid_davis.pkl
# ├── tapvid_kinetics
# │   ├── 0000_of_0010.pkl
# │   ├── 0001_of_0010.pkl
# │   ├── 0002_of_0010.pkl
# │   ├── 0003_of_0010.pkl
# │   ├── 0004_of_0010.pkl
# │   ├── 0005_of_0010.pkl
# │   ├── 0006_of_0010.pkl
# │   ├── 0007_of_0010.pkl
# │   ├── 0008_of_0010.pkl
# │   ├── 0009_of_0010.pkl
# │   ├── README.md
# │   ├── tapvid_kinetics.csv
# │   ├── test.txt
# │   ├── train.txt
# │   └── val.txt
# └── tapvid_rgb_stacking
#     ├── README.md
#     └── tapvid_rgb_stacking.pkl
```

## Visualize the Dataset

To visualize the dataset, you could run:
```bash
cd /path/to/pips
python3 -m pips_utils.visualize_tap \
  --input_path=<path_to_pickle_file.pkl> \
  --output_path=<path_to_output_video.mp4> \
  --alsologtostderr

# Examples:
python3 -m pips_utils.visualize_tapvid \
  --input_path=data/tapvid_davis/tapvid_davis.pkl \
  --output_path=logs/visualizations/tapvid_davis.mp4 \
  --alsologtostderr
python3 -m pips_utils.visualize_tapvid \
  --input_path=data/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl \
  --output_path=logs/visualizations/tapvid_rgb_stacking.mp4 \
  --alsologtostderr
python3 -m pips_utils.visualize_tapvid \
  --input_path=data/tapvid_kinetics/0009_of_0010.pkl \
  --output_path=logs/visualizations/tapvid_kinetics__0009_of_0010.mp4 \
  --alsologtostderr
```

For visualization examples, you can see the following:
- [TAP-Vid-DAVIS](https://storage.googleapis.com/dm-tapnet/content/davis_ground_truth_v2.html)
- [TAP-Vid-Kubric](https://storage.googleapis.com/dm-tapnet/content/kubric_ground_truth.html)
- [TAP-Vid-RGB-Stacking](https://storage.googleapis.com/dm-tapnet/content/rgb_stacking_ground_truth_v2.html)

## Evaluate models on TAP-Vid

Let's first evaluate RAFT on the dataset to try to reproducing some of
the numbers reported in the TAP-Vid paper.

```bash
python test_on_flt.py \
  --modeltype raft \
  --dataset_type tapvid \
  --dataset_location data/tapvid_davis/tapvid_davis.pkl \
  --exp_name tapvid_davis \
  --subset davis \
  --dataloader_workers 2
# 1_8_None_raft_tapvid_debug_17:24:03 step=000238/238 readtime=0.00 itertime=1.21 ate_all=15.88 ate_vis=12.79 ate_occ=130.60
# logs/1_8_None_raft_tapvid_debug_17:24:03/results_df.csv

python test_on_flt.py \
  --modeltype raft \
  --dataset_type tapvid \
  --dataset_location data/tapvid_kinetics \
  --exp_name tapvid_kinetics \
  --subset kinetics \
  --dataloader_workers 2

python test_on_flt.py \
  --modeltype raft \
  --dataset_type tapvid \
  --dataset_location data/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl \
  --exp_name tapvid_rgb_stacking \
  --subset kubric \
  --dataloader_workers 2
```

How about pips?

```bash
python test_on_flt.py \
  --modeltype pips \
  --dataset_type tapvid \
  --dataset_location data/tapvid_davis/tapvid_davis.pkl \
  --exp_name tapvid_davis \
  --subset davis \
  --dataloader_workers 2
# Results summary dataframe saved to:
# logs/1_8_None_pips_tapvid_davis_17:45:33/results_df.csv
# 
# TABLE: 'table2'
# | name   |     ade |   ade_visible |   ade_occluded |   ade_visible_chain |   ade_visible_chain_2 |   ade_visible_chain_4 |   ade_visible_chain_8 |   ade_mostly_visible |   ade_mostly_occluded |
# |:-------|--------:|--------------:|---------------:|--------------------:|----------------------:|----------------------:|----------------------:|---------------------:|----------------------:|
# | pips   | 12.4085 |       3.05335 |        205.119 |              2.9095 |               3.74382 |                4.8304 |               2.70431 |              10.8371 |               147.742 |

python test_on_flt.py \
  --modeltype pips \
  --dataset_type tapvid \
  --dataset_location data/tapvid_kinetics \
  --exp_name tapvid_kinetics \
  --subset kinetics \
  --dataloader_workers 2

python test_on_flt.py \
  --modeltype pips \
  --dataset_type tapvid \
  --dataset_location data/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl \
  --exp_name tapvid_rgb_stacking \
  --subset kubric \
  --dataloader_workers 2
```