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

### RAFT

The following setting reproduces the numbers reported in the TAP-Vid
paper. The numbers to not match exactly, i.e. 46 vs 48.

```bash
python -m evaluate \
  --modeltype raft \
  --dataset_type tapvid \
  --dataset_location data/tapvid_davis/tapvid_davis.pkl \
  --exp_name tapvid_davis \
  --subset davis \
  --query_mode strided
python -m evaluate \
  --modeltype raft \
  --dataset_type tapvid \
  --dataset_location data/tapvid_davis/tapvid_davis.pkl \
  --exp_name tapvid_davis \
  --subset davis \
  --query_mode first --log_freq 1


python -m pips_utils.figures \
  --mostly_visible_threshold 4 \
  --results_path_list \
  logs/1_8_None_raft_strided_tapvid_davis_2023.03.26_06.33.05/results_list.pkl \
  logs/1_8_None_raft_first_tapvid_davis_2023.03.26_06.32.39/results_list.pkl \
  --results_name_list \
  'DAVIS (strided)' \
  'DAVIS (first)'
# TABLE: 'table2-selected-metrics'
# |                           |   DAVIS (first) |   DAVIS (strided) |
# |:--------------------------|----------------:|------------------:|
# | ade                       |      102.796    |         62.8724   |
# | ade_visible               |       37.7445   |         27.7874   |
# | ade_occluded              |      215.751    |        196.908    |
# | ade_visible_chain         |       15.0625   |         11.0848   |
# | ade_visible_chain_2       |        0.72056  |          0.971356 |
# | ade_visible_chain_4       |        0.897851 |          2.34153  |
# | ade_visible_chain_8       |        6.48588  |          4.61155  |
# | pts_within_0.01           |        1.25612  |          2.04757  |
# | pts_within_0.1            |        2.83759  |          3.40933  |
# | pts_within_0.5            |       11.4767   |         12.4266   |
# | pts_within_1              |       17.7132   |         20.7424   |
# | pts_within_2              |       27.3157   |         32.7865   |
# | pts_within_4              |       40.1793   |         48.6673   |
# | pts_within_8              |       55.9183   |         64.7888   |
# | pts_within_16             |       67.3986   |         75.6063   |
# | average_pts_within_thresh |       41.705    |         48.5183   |
# 
# TABLE: 'table3-pck-metrics'
# |                           |   DAVIS (first) |   DAVIS (strided) |
# |:--------------------------|----------------:|------------------:|
# | ade_visible               |        33.7341  |          24.9943  |
# | ade_occluded              |       204.934   |         203.014   |
# | pts_within_0.01           |         1.14859 |           2.42119 |
# | pts_within_0.1            |         2.54648 |           4.02432 |
# | pts_within_0.5            |        10.6113  |          13.7736  |
# | pts_within_1              |        16.924   |          22.4812  |
# | pts_within_2              |        26.8288  |          35.2781  |
# | pts_within_4              |        40.8732  |          51.7743  |
# | pts_within_8              |        58.0549  |          67.4901  |
# | pts_within_16             |        70.5203  |          78.0162  |
# | average_pts_within_thresh |        42.6403  |          51.008   |



python -m evaluate \
  --modeltype raft \
  --dataset_type tapvid \
  --dataset_location data/tapvid_davis/tapvid_davis.pkl \
  --exp_name tapvid_kubric \
  --subset kubric \
  --query_mode strided
python -m evaluate \
  --modeltype raft \
  --dataset_type tapvid \
  --dataset_location data/tapvid_davis/tapvid_davis.pkl \
  --exp_name tapvid_kubric \
  --subset kubric \
  --query_mode first

python -m pips_utils.figures \
  --mostly_visible_threshold 4 \
  --results_path_list \
  logs/1_8_None_raft_tapvid_davis_2023.03.26_05.00.39/results_list.pkl \
  logs/1_8_None_raft_tapvid_davis_2023.03.26_05.00.27/results_list.pkl \
  --results_name_list \
  'KUBRIC (strided)' \
  'KUBRIC (first)'
# TODO Add numbers
```

### PIPs

TODO: Implement chained evaluation of PIP to allow for video sequences
      longer than 8 frames. Evaluate on TAP-Vid.

```bash
python -m evaluate \
  --modeltype pips \
  --dataset_type tapvid \
  --dataset_location data/tapvid_davis/tapvid_davis.pkl \
  --exp_name tapvid_davis \
  --subset davis \
  --query_mode strided
python -m evaluate \
  --modeltype pips \
  --dataset_type tapvid \
  --dataset_location data/tapvid_davis/tapvid_davis.pkl \
  --exp_name tapvid_davis \
  --subset davis \
  --query_mode first --log_freq 1


python -m pips_utils.figures \
  --mostly_visible_threshold 4 \
  --results_path_list \
  logs/1_8_None_raft_first_tapvid_davis_2023.03.26_06.32.39/results_list.pkl \
  logs/1_8_None_pips_first_tapvid_davis_2023.03.26_19.22.48/results_list.pkl \
  logs/1_8_None_pips_first_tapvid_davis_2023.03.26_19.46.35/results_list.pkl \
  --results_name_list \
  'RAFT DAVIS (first)' \
  'PIPS DAVIS (first)' \
  'PIPS+FeatInit DAVIS (first)'
