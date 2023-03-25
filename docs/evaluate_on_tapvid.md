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

### RAFT (stride of 5)

Let's first evaluate RAFT on the dataset to try to reproducing some of
the numbers reported in the TAP-Vid paper. TODO: The numbers do not
reproduce the paper numbers.

```bash
python test_on_flt.py \
  --modeltype raft \
  --dataset_type tapvid \
  --dataset_location data/tapvid_davis/tapvid_davis.pkl \
  --exp_name tapvid_davis \
  --subset davis \
  --dataloader_workers 2 \
  --S 5
# Results pickle file saved to:
# logs/1_5_None_raft_tapvid_davis_20:04:04/results_list.pkl
python -m pips_utils.figures --mostly_visible_threshold 4 --results_path_list logs/1_5_None_raft_tapvid_davis_22:17:43/results_list.pkl --results_name_list RAFT
# TABLE: 'table2-selected-metrics'
# |                           |      RAFT |
# |:--------------------------|----------:|
# | ade                       |   7.83185 |
# | ade_visible               |   1.92854 |
# | ade_occluded              | 228.308   |
# | ade_visible_chain         |   1.88262 |
# | ade_visible_chain_2       |   1.63043 |
# | ade_visible_chain_4       |   2.26388 |
# | ade_mostly_visible        |   3.98189 |
# | ade_mostly_occluded       | 113.99    |
# | pts_within_1              |  56.4134  |
# | pts_within_2              |  76.6075  |
# | pts_within_4              |  89.0857  |
# | pts_within_8              |  95.3939  |
# | pts_within_16             |  98.6494  |
# | average_pts_within_thresh |  83.23    |
# 
# TABLE: 'table3-tapvid-metrics-for-batched-df'
# |                           |      RAFT |
# |:--------------------------|----------:|
# | ade_visible               |   1.42952 |
# | ade_occluded              | 179.316   |
# | pts_within_0.01           |  20.585   |
# | pts_within_0.1            |  26.0005  |
# | pts_within_0.5            |  49.9805  |
# | pts_within_1              |  66.5964  |
# | pts_within_2              |  82.5936  |
# | pts_within_4              |  92.3043  |
# | pts_within_8              |  96.7841  |
# | pts_within_16             |  99.0649  |
# | average_pts_within_thresh |  87.4687  |


python test_on_flt.py \
  --modeltype raft \
  --dataset_type tapvid \
  --dataset_location data/tapvid_kinetics \
  --exp_name tapvid_kinetics \
  --subset kinetics \
  --dataloader_workers 2 \
  --S 5
# TODO: Dataloading needs to be rewritten, currently it is memory inefficient

# export CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
python test_on_flt.py \
  --modeltype raft \
  --dataset_type tapvid \
  --dataset_location data/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl \
  --exp_name tapvid_rgb_stacking \
  --subset kubric \
  --dataloader_workers 2 \
  --S 5
# Results pickle file saved to:
# logs/1_5_None_raft_tapvid_rgb_stacking_22:21:57/results_list.pkl
python -m pips_utils.figures --mostly_visible_threshold 4 --results_path_list logs/1_5_None_raft_tapvid_rgb_stacking_22:21:57/results_list.pkl --results_name_list RAFT
# TABLE: 'table2-selected-metrics'
# |                           |     RAFT |
# |:--------------------------|---------:|
# | ade                       |  1.38168 |
# | ade_visible               |  1.26206 |
# | ade_occluded              |  7.90721 |
# | ade_visible_chain         |  1.19539 |
# | ade_visible_chain_2       |  1.89112 |
# | ade_visible_chain_4       |  2.62895 |
# | ade_mostly_visible        |  1.26029 |
# | ade_mostly_occluded       |  5.26952 |
# | pts_within_1              | 77.234   |
# | pts_within_2              | 88.0266  |
# | pts_within_4              | 94.0605  |
# | pts_within_8              | 97.2683  |
# | pts_within_16             | 98.8774  |
# | average_pts_within_thresh | 91.0933  |
# 
# TABLE: 'table3-tapvid-metrics-for-batched-df'
# |                           |      RAFT |
# |:--------------------------|----------:|
# | ade_visible               |  0.954813 |
# | ade_occluded              |  7.32454  |
# | pts_within_0.01           | 20.6073   |
# | pts_within_0.1            | 31.6371   |
# | pts_within_0.5            | 68.3699   |
# | pts_within_1              | 82.676    |
# | pts_within_2              | 91.0452   |
# | pts_within_4              | 95.6189   |
# | pts_within_8              | 98.0127   |
# | pts_within_16             | 99.1844   |
# | average_pts_within_thresh | 93.3074   |
```

### RAFT (stride of 8)

```bash
python test_on_flt.py \
  --modeltype raft \
  --dataset_type tapvid \
  --dataset_location data/tapvid_davis/tapvid_davis.pkl \
  --exp_name tapvid_davis \
  --subset davis \
  --dataloader_workers 2
# Results pickle file saved to:
# logs/1_8_None_raft_tapvid_davis_17:11:38/results_list.pkl
# 
# Results summary dataframe saved to:
# logs/1_8_None_raft_tapvid_davis_17:11:38/results_df.csv
# 
# TABLE: 'table2'
# |                           |       raft |
# |:--------------------------|-----------:|
# | ade                       |  12.6258   |
# | ade_visible               |   3.11438  |
# | ade_occluded              | 206.734    |
# | ade_visible_chain         |   2.84771  |
# | ade_visible_chain_2       |   2.7475   |
# | ade_visible_chain_4       |   2.99066  |
# | ade_visible_chain_8       |   2.79838  |
# | ade_mostly_visible        |  10.7497   |
# | ade_mostly_occluded       | 150.457    |
# | pts_within_1              |  45.1254   |
# | pts_within_2              |  66.8288   |
# | pts_within_4              |  83.5351   |
# | pts_within_8              |  91.6531   |
# | pts_within_16             |  96.368    |
# | average_pts_within_thresh |  76.7021   |

python test_on_flt.py \
  --modeltype raft \
  --dataset_type tapvid \
  --dataset_location data/tapvid_kinetics \
  --exp_name tapvid_kinetics \
  --subset kinetics \
  --dataloader_workers 2
# TODO: run

python test_on_flt.py \
  --modeltype raft \
  --dataset_type tapvid \
  --dataset_location data/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl \
  --exp_name tapvid_rgb_stacking \
  --subset kubric \
  --dataloader_workers 2
# 1_8_None_raft_tapvid_rgb_stacking_14:06:31 step=000750/750 readtime=0.00 itertime=4.17 ate_all=2.02 ate_vis=1.85 ate_occ=7.38
#
# Results pickle file saved to:
# logs/1_8_None_raft_tapvid_rgb_stacking_14:06:31/results_list.pkl
#
# Results summary dataframe saved to:
# logs/1_8_None_raft_tapvid_rgb_stacking_14:06:31/results_df.csv
#
# TABLE: 'table2'
# |                           |     raft |
# |:--------------------------|---------:|
# | ade                       |  2.2244  |
# | ade_visible               |  1.94877 |
# | ade_occluded              | 10.3683  |
# | ade_visible_chain         |  1.6206  |
# | ade_visible_chain_2       |  1.92072 |
# | ade_visible_chain_4       |  2.31691 |
# | ade_visible_chain_8       |  1.50446 |
# | ade_mostly_visible        |  2.1193  |
# | ade_mostly_occluded       |  8.43689 |
# | pts_within_1              | 68.9905  |
# | pts_within_2              | 83.43    |
# | pts_within_4              | 91.3734  |
# | pts_within_8              | 95.7123  |
# | pts_within_16             | 97.8531  |
# | average_pts_within_thresh | 87.4719  |
python -m pips_utils.figures --mostly_visible_threshold 4 --results_path_list logs/1_8_None_raft_tapvid_rgb_stacking_14:06:31/results_list.pkl --results_name_list RAFT
```

### PIPs (stride of 8)

Evaluating the PIPs checkpoint that takes exactly 8 frames as input. I
will therefore simply use a stride of 8. To use different strides, one
could (i) train a new model for that stride, (ii) use chaining for
longer sequences, (iii) drop the tail predictions for shorter sequences
(provided sequences of 8 frames are provided as input), (iv) invent an
architecture/setup that takes an arbitrary number of frames.

```bash
python test_on_flt.py \
  --modeltype pips \
  --dataset_type tapvid \
  --dataset_location data/tapvid_davis/tapvid_davis.pkl \
  --exp_name tapvid_davis \
  --subset davis \
  --dataloader_workers 2
# Results pickle file saved to:
# logs/1_8_None_pips_tapvid_davis_16:46:24/results_list.pkl
#
# Results summary dataframe saved to:
# logs/1_8_None_pips_tapvid_davis_16:46:24/results_df.csv
#
# TABLE: 'table2'
# |                           |       PIPs |
# |:--------------------------|-----------:|
# | ade                       |  11.7334   |
# | ade_visible               |   2.14757  |
# | ade_occluded              | 207.592    |
# | ade_visible_chain         |   2.03914  |
# | ade_visible_chain_2       |   2.31194  |
# | ade_visible_chain_4       |   3.20572  |
# | ade_visible_chain_8       |   1.92732  |
# | ade_mostly_visible        |   9.86985  |
# | ade_mostly_occluded       | 150.411    |
# | pts_within_1              |  47.2241   |
# | pts_within_2              |  74.9366   |
# | pts_within_4              |  90.5042   |
# | pts_within_8              |  96.1684   |
# | pts_within_16             |  98.3772   |
# | average_pts_within_thresh |  81.4421   |

python test_on_flt.py \
  --modeltype pips \
  --dataset_type tapvid \
  --dataset_location data/tapvid_kinetics \
  --exp_name tapvid_kinetics \
  --subset kinetics \
  --dataloader_workers 2
# TODO: run

python test_on_flt.py \
  --modeltype pips \
  --dataset_type tapvid \
  --dataset_location data/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl \
  --exp_name tapvid_rgb_stacking \
  --subset kubric \
  --dataloader_workers 2
# 1_8_None_pips_tapvid_rgb_stacking_14:05:59 step=000750/750 readtime=0.00 itertime=7.21 ate_all=1.76 ate_vis=1.62 ate_occ=6.41
#
# Results pickle file saved to:
# logs/1_8_None_pips_tapvid_rgb_stacking_14:05:59/results_list.pkl
#
# Results summary dataframe saved to:
# logs/1_8_None_pips_tapvid_rgb_stacking_14:05:59/results_df.csv
#
# TABLE: 'table2'
# |                           |     pips |
# |:--------------------------|---------:|
# | ade                       |  1.94162 |
# | ade_visible               |  1.70433 |
# | ade_occluded              |  8.24842 |
# | ade_visible_chain         |  1.63541 |
# | ade_visible_chain_2       |  2.39923 |
# | ade_visible_chain_4       |  2.60406 |
# | ade_visible_chain_8       |  1.49852 |
# | ade_mostly_visible        |  1.85328 |
# | ade_mostly_occluded       |  7.32662 |
# | pts_within_1              | 61.3769  |
# | pts_within_2              | 82.4177  |
# | pts_within_4              | 92.4851  |
# | pts_within_8              | 96.9728  |
# | pts_within_16             | 98.8577  |
# | average_pts_within_thresh | 86.4221  |
python -m pips_utils.figures --mostly_visible_threshold 4 --results_path_list logs/1_8_None_pips_tapvid_rgb_stacking_14:05:59/results_list.pkl --results_name_list RAFT
```

### Compare models (stride 8)

```bash
# Davis
python -m pips_utils.figures \
  --mostly_visible_threshold 4 \
  --results_path_list \
  logs/1_8_None_raft_tapvid_davis_17:11:38/results_list.pkl \
  logs/1_8_None_pips_tapvid_davis_16:46:24/results_list.pkl \
  --results_name_list \
  RAFT \
  PIPs
# TABLE: 'table2-selected-metrics'
# |                           |      RAFT |      PIPs |
# |:--------------------------|----------:|----------:|
# | ade                       |  12.6258  |  11.7334  |
# | ade_visible               |   3.11438 |   2.14757 |
# | ade_occluded              | 206.734   | 207.592   |
# | ade_visible_chain         |   2.84771 |   2.03914 |
# | ade_visible_chain_2       |   2.7475  |   2.31194 |
# | ade_visible_chain_4       |   2.99066 |   3.20572 |
# | ade_visible_chain_8       |   2.79838 |   1.92732 |
# | ade_mostly_visible        |  10.7497  |   9.86985 |
# | ade_mostly_occluded       | 150.457   | 150.411   |
# | pts_within_1              |  45.1254  |  47.2241  |
# | pts_within_2              |  66.8288  |  74.9366  |
# | pts_within_4              |  83.5351  |  90.5042  |
# | pts_within_8              |  91.6531  |  96.1684  |
# | pts_within_16             |  96.368   |  98.3772  |
# | average_pts_within_thresh |  76.7021  |  81.4421  |
# 
# TABLE: 'table3-tapvid-metrics-for-batched-df'
# |                           |      RAFT |      PIPs |
# |:--------------------------|----------:|----------:|
# | ade_visible               |   2.62461 |   1.79725 |
# | ade_occluded              | 185.274   | 185.455   |
# | pts_within_0.01           |  13.1581  |  13.1124  |
# | pts_within_0.1            |  16.9583  |  14.0635  |
# | pts_within_0.5            |  36.374   |  30.6984  |
# | pts_within_1              |  52.2824  |  54.6219  |
# | pts_within_2              |  71.3025  |  78.7964  |
# | pts_within_4              |  86.199   |  92.2094  |
# | pts_within_8              |  93.1147  |  96.9796  |
# | pts_within_16             |  97.0062  |  98.7066  |
# | average_pts_within_thresh |  79.981   |  84.2628  |

# Kinetics
TODO: run

# Kubric
  python -m pips_utils.figures \
    --mostly_visible_threshold 4 \
    --results_path_list \
    logs/1_8_None_raft_tapvid_rgb_stacking_14:06:31/results_list.pkl \
    logs/1_8_None_pips_tapvid_rgb_stacking_14:05:59/results_list.pkl \
    --results_name_list \
    RAFT \
    PIPs
# TABLE: 'table2-selected-metrics'
# |                           |     RAFT |     PIPs |
# |:--------------------------|---------:|---------:|
# | ade                       |  2.2244  |  1.94162 |
# | ade_visible               |  1.94877 |  1.70433 |
# | ade_occluded              | 10.3683  |  8.24842 |
# | ade_visible_chain         |  1.6206  |  1.63541 |
# | ade_visible_chain_2       |  1.92072 |  2.39923 |
# | ade_visible_chain_4       |  2.31691 |  2.60406 |
# | ade_visible_chain_8       |  1.50446 |  1.49852 |
# | ade_mostly_visible        |  2.1193  |  1.85328 |
# | ade_mostly_occluded       |  8.43689 |  7.32662 |
# | pts_within_1              | 68.9905  | 61.3769  |
# | pts_within_2              | 83.43    | 82.4177  |
# | pts_within_4              | 91.3734  | 92.4851  |
# | pts_within_8              | 95.7123  | 96.9728  |
# | pts_within_16             | 97.8531  | 98.8577  |
# | average_pts_within_thresh | 87.4719  | 86.4221  |
# 
# TABLE: 'table3-tapvid-metrics-for-batched-df'
# |                           |     RAFT |     PIPs |
# |:--------------------------|---------:|---------:|
# | ade_visible               |  1.58314 |  1.41779 |
# | ade_occluded              | 10.5757  |  8.42246 |
# | pts_within_0.01           | 13.1374  | 13.0203  |
# | pts_within_0.1            | 21.3299  | 14.7026  |
# | pts_within_0.5            | 55.6176  | 41.4273  |
# | pts_within_1              | 73.7282  | 67.4319  |
# | pts_within_2              | 86.2863  | 85.5612  |
# | pts_within_4              | 93.048   | 93.9517  |
# | pts_within_8              | 96.6369  | 97.6018  |
# | pts_within_16             | 98.346   | 99.0976  |
# | average_pts_within_thresh | 89.6091  | 88.7288  |
```
