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

Run the evaluations (i) for all 4 TAP-Vid dataset subsets (DAVIS,
KUBRIC, RGB-STACKING, KINETICS), (ii) for RAFT and PIPS, (iii) for
"strided" and "first" query mode.

```bash
# DAVIS (30 batches)
python -m evaluate --modeltype raft --dataset_type tapvid --dataset_location data/tapvid_davis/tapvid_davis.pkl --subset davis --query_mode strided
python -m evaluate --modeltype raft --dataset_type tapvid --dataset_location data/tapvid_davis/tapvid_davis.pkl --subset davis --query_mode first   --log_freq 1
python -m evaluate --modeltype pips --dataset_type tapvid --dataset_location data/tapvid_davis/tapvid_davis.pkl --subset davis --query_mode strided
python -m evaluate --modeltype pips --dataset_type tapvid --dataset_location data/tapvid_davis/tapvid_davis.pkl --subset davis --query_mode first   --log_freq 1

# KUBRIC (250 batches)
# export CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
python -m evaluate --modeltype raft --dataset_type tapvid --dataset_location none --subset kubric --query_mode strided
python -m evaluate --modeltype raft --dataset_type tapvid --dataset_location none --subset kubric --query_mode first  
python -m evaluate --modeltype pips --dataset_type tapvid --dataset_location none --subset kubric --query_mode strided
python -m evaluate --modeltype pips --dataset_type tapvid --dataset_location none --subset kubric --query_mode first  

# RGB-STACKING (50 batches)
python -m evaluate --modeltype raft --dataset_type tapvid --dataset_location data/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl --subset rgb_stacking --query_mode strided --dont_save_raw_results --no_visualisations
python -m evaluate --modeltype raft --dataset_type tapvid --dataset_location data/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl --subset rgb_stacking --query_mode first  
python -m evaluate --modeltype pips --dataset_type tapvid --dataset_location data/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl --subset rgb_stacking --query_mode strided --dont_save_raw_results --no_visualisations
python -m evaluate --modeltype pips --dataset_type tapvid --dataset_location data/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl --subset rgb_stacking --query_mode first  

# KINETICS (1144 batches)
python -m evaluate --modeltype raft --dataset_type tapvid --dataset_location data/tapvid_kinetics --subset kinetics --query_mode strided --dont_save_raw_results --no_visualisations
python -m evaluate --modeltype raft --dataset_type tapvid --dataset_location data/tapvid_kinetics --subset kinetics --query_mode first  
python -m evaluate --modeltype pips --dataset_type tapvid --dataset_location data/tapvid_kinetics --subset kinetics --query_mode strided --dont_save_raw_results --no_visualisations
python -m evaluate --modeltype pips --dataset_type tapvid --dataset_location data/tapvid_kinetics --subset kinetics --query_mode first  
```

Digest the results with tables and figures:

```bash
ls -hail logs/*/*.pkl

# 105949204 -rw-r--r-- 1 rajic sc-pme  19M Mar 28 21:50 logs/1_8_None_pips_72_tapvid_davis_first_2023.03.28_21.46.45/results_list.pkl
#   7042148 -rw-r--r-- 1 rajic sc-pme 6.9G Mar 29 03:31 logs/1_8_None_pips_72_tapvid_kubric_first_2023.03.29_02.46.25/results_list.pkl
# 105949342 -rw-r--r-- 1 rajic sc-pme 282M Mar 29 03:08 logs/1_8_None_pips_72_tapvid_rgb_stacking_first_2023.03.29_02.46.09/results_list.pkl
#   7042150 -rw-r--r-- 1 rajic sc-pme 5.7G Mar 29 11:17 logs/1_8_None_pips_72_tapvid_kinetics_first_2023.03.28_22.18.09/results_list.pkl

#   7042136 -rw-r--r-- 1 rajic sc-pme 1.9G Mar 28 21:42 logs/1_8_None_pips_72_tapvid_davis_strided_2023.03.28_21.30.43/results_list.pkl
# 112983237 -rw-r--r-- 1 rajic sc-pme 6.9G Mar 28 22:45 logs/1_8_None_pips_72_tapvid_kubric_strided_2023.03.28_22.22.20/results_list.pkl
# 112983258 -rw-r--r-- 1 rajic sc-pme  31M Mar 29 03:54 logs/1_8_None_pips_72_tapvid_rgb_stacking_strided_2023.03.29_02.43.37/results_df.csv
# 109441907 -rw-r--r-- 1 rajic sc-pme 490M Mar 30 21:21 logs/1_8_None_pips_72_tapvid_kinetics_strided_2023.03.29_13.53.06/results_df.csv

# 111928232 -rw-r--r-- 1 rajic sc-pme  26M Mar 27 05:07 logs/1_8_None_raft_72_tapvid_davis_first_2023.03.27_04.44.23/results_list.pkl
# 105344197 -rw-r--r-- 1 rajic sc-pme 6.9G Mar 27 13:02 logs/1_8_None_raft_72_tapvid_kubric_first_2023.03.27_12.25.42/results_list.pkl
#    434597 -rw-r--r-- 1 rajic sc-pme 367M Mar 27 05:51 logs/1_8_None_raft_72_tapvid_rgb_stacking_first_2023.03.27_04.45.22/results_list.pkl
# 111493240 -rw-r--r-- 1 rajic sc-pme 7.4G Mar 28 07:22 logs/1_8_None_raft_72_tapvid_kinetics_first_2023.03.27_05.17.15/results_list.pkl

#   5987026 -rw-r--r-- 1 rajic sc-pme 2.6G Mar 27 04:57 logs/1_8_None_raft_72_tapvid_davis_strided_2023.03.27_04.43.43/results_list.pkl
# 111928218 -rw-r--r-- 1 rajic sc-pme 6.9G Mar 27 05:14 logs/1_8_None_raft_72_tapvid_kubric_strided_2023.03.27_04.45.10/results_list.pkl
# 110391004 -rw-r--r-- 1 rajic sc-pme  33M Mar 29 04:14 logs/1_8_None_raft_72_tapvid_rgb_stacking_strided_2023.03.29_02.43.18/results_df.csv
    # 13323 -rw-r--r-- 1 rajic sc-pme 502M Mar 30 21:21 logs/1_8_None_raft_72_tapvid_kinetics_strided_2023.03.29_13.53.06/results_df.csv

# Strided query mode
python -m pips_utils.figures --mostly_visible_threshold 4 --results_path_list \
  logs/1_8_None_pips_72_tapvid_davis_strided_2023.03.28_21.30.43/results_list.pkl \
  logs/1_8_None_raft_72_tapvid_davis_strided_2023.03.27_04.43.43/results_list.pkl \
  logs/1_8_None_pips_72_tapvid_kubric_strided_2023.03.28_22.22.20/results_list.pkl \
  logs/1_8_None_raft_72_tapvid_kubric_strided_2023.03.27_04.45.10/results_list.pkl \
  logs/1_8_None_pips_72_tapvid_rgb_stacking_strided_2023.03.29_02.43.37/results_df.csv \
  logs/1_8_None_raft_72_tapvid_rgb_stacking_strided_2023.03.29_02.43.18/results_df.csv \
  logs/1_8_None_pips_72_tapvid_kinetics_strided_2023.03.29_13.53.06/results_df.csv \
  logs/1_8_None_raft_72_tapvid_kinetics_strided_2023.03.29_13.53.06/results_df.csv
# TABLE: 'table3-pck-metrics'
# |                           |   ('0__pips_tapvid_davis', 'DAVIS', 'pips-8') |   ('1__raft_tapvid_davis', 'DAVIS', 'raft') |   ('2__pips_tapvid_kubric', 'KUBRIC', 'pips-8') |   ('3__raft_tapvid_kubric', 'KUBRIC', 'raft') |   ('4__pips_tapvid_rgb-stacking', 'RGB-STACKING', 'pips-8') |   ('5__raft_tapvid_rgb-stacking', 'RGB-STACKING', 'raft') |   ('6__pips_tapvid_kinetics', 'KINETICS', 'pips-8') |   ('7__raft_tapvid_kinetics', 'KINETICS', 'raft') |
# |:--------------------------|----------------------------------------------:|--------------------------------------------:|------------------------------------------------:|----------------------------------------------:|------------------------------------------------------------:|----------------------------------------------------------:|----------------------------------------------------:|--------------------------------------------------:|
# | ade                       |                                   47.549      |                                  68.5299    |                                       7.87962   |                                    12.221     |                                                  8.75234    |                                                10.5497    |                                         61.6588     |                                        79.9767    |
# | ade_visible               |                                    9.0368     |                                  24.9943    |                                       3.42312   |                                     5.50041   |                                                  7.44406    |                                                 9.49906   |                                         10.2739     |                                        18.7618    |
# | ade_occluded              |                                  174.953      |                                 203.014     |                                      21.3145    |                                    32.6482    |                                                 19.687      |                                                22.5006    |                                        191.16       |                                       217.79      |
# | ade_visible_chain         |                                    4.77485    |                                   9.94658   |                                       2.6285    |                                     2.90174   |                                                  5.51253    |                                                 5.32374   |                                          5.49815    |                                         8.10727   |
# | jaccard_1                 |                                   11.4328     |                                  11.1023    |                                      21.7327    |                                    31.4773    |                                                  7.38147    |                                                11.1943    |                                          7.55515    |                                        10.754     |
# | jaccard_2                 |                                   26.6053     |                                  20.0358    |                                      51.4988    |                                    54.4946    |                                                 18.0966     |                                                22.6721    |                                         18.602      |                                        21.7157    |
# | jaccard_4                 |                                   47.809      |                                  33.7822    |                                      70.5513    |                                    69.2909    |                                                 38.3963     |                                                41.6548    |                                         38.0347     |                                        37.6201    |
# | jaccard_8                 |                                   63.6225     |                                  49.4162    |                                      79.0621    |                                    77.1541    |                                                 61.1227     |                                                63.3421    |                                         55.4265     |                                        52.7481    |
# | jaccard_16                |                                   71.1309     |                                  60.9399    |                                      82.8896    |                                    81.256     |                                                 78.8077     |                                                76.523     |                                         63.4184     |                                        62.2318    |
# | pts_within_0.01           |                                    0.00436136 |                                   0.0295142 |                                       0.0043016 |                                     0.0272686 |                                                  0.00317241 |                                                 0.0298195 |                                          0.00365239 |                                         0.0260607 |
# | pts_within_0.1            |                                    0.391393   |                                   1.64864   |                                       0.386806  |                                     1.51934   |                                                  0.300302   |                                                 1.62266   |                                          0.355335   |                                         1.37153   |
# | pts_within_0.5            |                                    8.18957    |                                  11.5977    |                                       8.99222   |                                    12.9509    |                                                  5.30708    |                                                 9.97611   |                                          6.45527    |                                        10.0257    |
# | pts_within_1              |                                   21.6376     |                                  20.5219    |                                      32.2758    |                                    42.6751    |                                                 13.1963     |                                                19.3408    |                                         16.4047     |                                        20.0463    |
# | pts_within_2              |                                   42.9702     |                                  33.6256    |                                      64.8038    |                                    66.4829    |                                                 28.2059     |                                                34.4954    |                                         34.7734     |                                        35.5904    |
# | pts_within_4              |                                   66.7702     |                                  50.5182    |                                      83.4017    |                                    80.661     |                                                 51.2294     |                                                54.7771    |                                         59.8807     |                                        54.3876    |
# | pts_within_8              |                                   82.1751     |                                  66.5963    |                                      91.9835    |                                    88.3082    |                                                 73.3641     |                                                74.2196    |                                         78.931      |                                        70.3155    |
# | pts_within_16             |                                   89.5509     |                                  77.3434    |                                      96.2266    |                                    92.512     |                                                 89.1132     |                                                85.7356    |                                         87.7056     |                                        80.2374    |
# | average_jaccard           |                                   44.1201     |                                  35.0553    |                                      61.1469    |                                    62.7346    |                                                 40.761      |                                                43.0773    |                                         36.6073     |                                        37.0139    |
# | average_pts_within_thresh |                                   60.6208     |                                  49.7211    |                                      73.7383    |                                    74.1279    |                                                 51.0218     |                                                53.7137    |                                         55.5391     |                                        52.1154    |
# | occlusion_accuracy        |                                   82.0658     |                                  79.7149    |                                      87.7869    |                                    88.963     |                                                 91.5168     |                                                90.691     |                                         76.5936     |                                        80.4086    |
# 
# Done. Figures saved to: logs/figures/2023.03.31_02.58.56


# First query mode
python -m pips_utils.figures --mostly_visible_threshold 4 --results_path_list \
  logs/1_8_None_pips_72_tapvid_davis_first_2023.03.28_21.46.45/results_df.csv \
  logs/1_8_None_raft_72_tapvid_davis_first_2023.03.27_04.44.23/results_df.csv \
  logs/1_8_None_pips_72_tapvid_kubric_first_2023.03.29_02.46.25/results_df.csv \
  logs/1_8_None_raft_72_tapvid_kubric_first_2023.03.27_12.25.42/results_df.csv \
  logs/1_8_None_pips_72_tapvid_rgb_stacking_first_2023.03.29_02.46.09/results_df.csv \
  logs/1_8_None_raft_72_tapvid_rgb_stacking_first_2023.03.27_04.45.22/results_df.csv \
  logs/1_8_None_pips_72_tapvid_kinetics_first_2023.03.28_22.18.09/results_df.csv \
  logs/1_8_None_raft_72_tapvid_kinetics_first_2023.03.27_05.17.15/results_df.csv
# TABLE: 'table3-pck-metrics'
# |                           |   ('0__pips_tapvid_davis', 'DAVIS', 'pips-8') |   ('1__raft_tapvid_davis', 'DAVIS', 'raft') |   ('2__pips_tapvid_kubric', 'KUBRIC', 'pips-8') |   ('3__raft_tapvid_kubric', 'KUBRIC', 'raft') |   ('4__pips_tapvid_rgb-stacking', 'RGB-STACKING', 'pips-8') |   ('5__raft_tapvid_rgb-stacking', 'RGB-STACKING', 'raft') |   ('6__pips_tapvid_kinetics', 'KINETICS', 'pips-8') |   ('7__raft_tapvid_kinetics', 'KINETICS', 'raft') |
# |:--------------------------|----------------------------------------------:|--------------------------------------------:|------------------------------------------------:|----------------------------------------------:|------------------------------------------------------------:|----------------------------------------------------------:|----------------------------------------------------:|--------------------------------------------------:|
# | ade                       |                                    62.2934    |                                92.8603      |                                      8.36493    |                                   9.42237     |                                                 13.5075     |                                              16.8058      |                                          84.0813    |                                     110.072       |
# | ade_visible               |                                    10.0598    |                                33.7341      |                                      3.43031    |                                   5.44369     |                                                  9.3707     |                                              13.2476      |                                          13.4395    |                                      25.1589      |
# | ade_occluded              |                                   176.917     |                               204.934       |                                     21.5205     |                                  21.3207      |                                                 24.1837     |                                              28.6165      |                                         191.153     |                                     224.167       |
# | ade_visible_chain         |                                     5.07903   |                                13.9067      |                                      2.66527    |                                   2.8587      |                                                  5.68421    |                                               6.28541     |                                           6.77118   |                                      10.574       |
# | jaccard_1                 |                                     7.80869   |                                 0           |                                     21.5674     |                                   0.000302864 |                                                  4.35928    |                                               6.03318e-06 |                                           4.17981   |                                       3.4176e-05  |
# | jaccard_2                 |                                    18.9297    |                                 0           |                                     51.4366     |                                   0.000609798 |                                                 11.7461     |                                               0.000104069 |                                          11.3905    |                                       0.000138387 |
# | jaccard_4                 |                                    38.2839    |                                 5.42005e-05 |                                     70.4723     |                                   0.00101485  |                                                 28.091      |                                               0.000445363 |                                          26.4582    |                                       0.000553951 |
# | jaccard_8                 |                                    54.0613    |                                 0.00017094  |                                     79.0582     |                                   0.00141529  |                                                 49.4893     |                                               0.00124182  |                                          42.7053    |                                       0.00187543  |
# | jaccard_16                |                                    63.0758    |                                 0.000197084 |                                     82.8288     |                                   0.00186607  |                                                 66.2024     |                                               0.00402867  |                                          51.0454    |                                       0.00422485  |
# | pts_within_0.01           |                                     0.0056046 |                                 0.0381538   |                                      0.00327896 |                                   0.0250379   |                                                  0.00191473 |                                               0.0212816   |                                           0.0025178 |                                       0.015255    |
# | pts_within_0.1            |                                     0.359909  |                                 1.43982     |                                      0.394817   |                                   1.50931     |                                                  0.202257   |                                               1.02691     |                                           0.221019  |                                       0.779321    |
# | pts_within_0.5            |                                     6.31902   |                                 9.5823      |                                      8.93374    |                                  12.9941      |                                                  3.88683    |                                               6.52628     |                                           4.16655   |                                       6.51279     |
# | pts_within_1              |                                    17.2526    |                                16.0198      |                                     32.1551     |                                  42.7254      |                                                 10.1675     |                                              12.9908      |                                          11.2211    |                                      13.9224      |
# | pts_within_2              |                                    35.3658    |                                26.176       |                                     64.8381     |                                  66.5947      |                                                 22.9047     |                                              24.8304      |                                          25.9778    |                                      26.6822      |
# | pts_within_4              |                                    59.6411    |                                40.3949      |                                     83.4129     |                                  80.7931      |                                                 44.3564     |                                              42.9396      |                                          49.5846    |                                      44.2573      |
# | pts_within_8              |                                    77.0272    |                                57.7502      |                                     92.0408     |                                  88.451       |                                                 67.5614     |                                              63.3213      |                                          71.2775    |                                      62.0621      |
# | pts_within_16             |                                    87.101     |                                70.3036      |                                     96.1774     |                                  92.6602      |                                                 84.4682     |                                              78.6077      |                                          82.738     |                                      74.5001      |
# | average_jaccard           |                                    36.4319    |                                 8.44449e-05 |                                     61.0727     |                                   0.00104177  |                                                 31.9776     |                                               0.00116519  |                                          27.1558    |                                       0.00136536  |
# | average_pts_within_thresh |                                    55.2775    |                                42.1289      |                                     73.7249     |                                  74.2449      |                                                 45.8916     |                                              44.538       |                                          48.1598    |                                      44.2848      |
# | occlusion_accuracy        |                                    76.1238    |                                 0.273583    |                                     87.7514     |                                   0.110541    |                                                 83.3483     |                                               0.158505    |                                          67.4002    |                                       0.257191    |
# 
# Done. Figures saved to: logs/figures/2023.03.31_03.27.19
```

## Visualize Trajectories to Identify Failure Cases

Run the evaluation for the first 3 scenes and find the visualisations in tensorboard logs:
```bash
python -m evaluate --modeltype pips --dataset_type tapvid --dataset_location data/tapvid_davis/tapvid_davis.pkl --subset davis --query_mode first --log_freq 1 --max_iter 3
```

## Training on Kubric

Training from scratch with three different learning rate settings:
```bash
# FLT++
python train.py --max_iters 200000 --B=1 --lr 5e-4

# export CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
python train.py --dataset_type tapvid-chunked --dataset_location data/tapvid_kubric --subset_train kubric-train --subset_valid kubric --max_iters 200000 --B=1 --lr 5e-5
python train.py --dataset_type tapvid-chunked --dataset_location data/tapvid_kubric --subset_train kubric-train --subset_valid kubric --max_iters 200000 --B=1 --lr 5e-4
python train.py --dataset_type tapvid-chunked --dataset_location data/tapvid_kubric --subset_train kubric-train --subset_valid kubric --max_iters 200000 --B=1 --lr 5e-3

# logs/my_pips/checkpoints/1_8_768_72_tapvid-chunked_train=kubric-train_val=kubric4hv_I4_5e-5_A_debug_2023.04.02_15.07.04
# logs/my_pips/checkpoints/1_8_768_72_tapvid-chunked_train=kubric-train_val=kubric4hv_I4_5e-4_A_debug_2023.04.02_15.06.46
# logs/my_pips/checkpoints/1_8_768_72_tapvid-chunked_train=kubric-train_val=kubric4hv_I4_5e-3_A_debug_2023.04.02_15.06.47

python -m evaluate --modeltype pips --checkpoint_path logs/my_pips/checkpoints/1_8_768_72_tapvid-chunked_train=kubric-train_val=kubric4hv_I4_5e-5_A_debug_2023.04.02_15.07.04 --dataset_type tapvid --dataset_location data/tapvid_davis/tapvid_davis.pkl --subset davis --query_mode first
python -m evaluate --modeltype pips --checkpoint_path logs/my_pips/checkpoints/1_8_768_72_tapvid-chunked_train=kubric-train_val=kubric4hv_I4_5e-5_A_debug_2023.04.02_15.07.04 --dataset_type tapvid --dataset_location none --subset kubric --query_mode first  
python -m evaluate --modeltype pips --checkpoint_path logs/my_pips/checkpoints/1_8_768_72_tapvid-chunked_train=kubric-train_val=kubric4hv_I4_5e-5_A_debug_2023.04.02_15.07.04 --dataset_type tapvid --dataset_location data/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl --subset rgb_stacking --query_mode first  
python -m evaluate --modeltype pips --checkpoint_path logs/my_pips/checkpoints/1_8_768_72_tapvid-chunked_train=kubric-train_val=kubric4hv_I4_5e-5_A_debug_2023.04.02_15.07.04 --dataset_type tapvid --dataset_location data/tapvid_kinetics --subset kinetics --query_mode first  
# 1_8_None_pips_72_tapvid_davis_first_2023.04.03_09.29.24
# 1_8_None_pips_72_tapvid_kubric_first_2023.04.03_09.29.28
# 1_8_None_pips_72_tapvid_rgb_stacking_first_2023.04.03_09.29.33
# 1_8_None_pips_72_tapvid_kinetics_first_2023.04.03_09.29.37
python -m pips_utils.figures --mostly_visible_threshold 4 --results_path_list \
  logs/1_8_None_pips_72_tapvid_davis_first_2023.04.03_09.29.24/results_df.csv \
  logs/1_8_None_pips_72_tapvid_kubric_first_2023.04.03_09.29.28/results_df.csv \
  logs/1_8_None_pips_72_tapvid_rgb_stacking_first_2023.04.03_09.29.33/results_df.csv
# TABLE: 'table3-pck-metrics'
# |                           |   ('0__pips_tapvid_davis', 'DAVIS', 'pips-8') |   ('1__pips_tapvid_kubric', 'KUBRIC', 'pips-8') |   ('2__pips_tapvid_rgb-stacking', 'RGB-STACKING', 'pips-8') |
# |:--------------------------|----------------------------------------------:|------------------------------------------------:|------------------------------------------------------------:|
# | ade                       |                                     66.8994   |                                      9.18836    |                                                 17.8319     |
# | ade_visible               |                                     14.9606   |                                      4.8918     |                                                 12.9553     |
# | ade_occluded              |                                    177.484    |                                     22.4812     |                                                 29.4196     |
# | ade_visible_chain         |                                      7.83264  |                                      3.81523    |                                                  8.56763    |
# | jaccard_1                 |                                      7.48129  |                                     25.3345     |                                                  4.13305    |
# | jaccard_2                 |                                     17.1349   |                                     49.7926     |                                                  9.76714    |
# | jaccard_4                 |                                     32.0398   |                                     66.7062     |                                                 21.2501     |
# | jaccard_8                 |                                     45.4194   |                                     75.8929     |                                                 37.9763     |
# | jaccard_16                |                                     53.275    |                                     80.8137     |                                                 56.4692     |
# | pts_within_0.01           |                                      0        |                                      0.00632069 |                                                  0.00382214 |
# | pts_within_0.1            |                                      0.328671 |                                      0.570098   |                                                  0.406061   |
# | pts_within_0.5            |                                      5.58236  |                                     10.3824     |                                                  4.01933    |
# | pts_within_1              |                                     14.5136   |                                     35.2263     |                                                  9.20824    |
# | pts_within_2              |                                     29.7198   |                                     60.5215     |                                                 18.7458     |
# | pts_within_4              |                                     49.8224   |                                     77.1815     |                                                 34.7063     |
# | pts_within_8              |                                     65.4831   |                                     86.9984     |                                                 54.3838     |
# | pts_within_16             |                                     75.7181   |                                     92.9072     |                                                 74.3588     |
# | average_jaccard           |                                     31.0701   |                                     59.708      |                                                 25.9192     |
# | average_pts_within_thresh |                                     47.0514   |                                     70.567      |                                                 38.2806     |
# | occlusion_accuracy        |                                     76.1739   |                                     88.5859     |                                                 83.1172     |

python -m evaluate --modeltype pips --checkpoint_path logs/my_pips/checkpoints/1_8_768_72_tapvid-chunked_train=kubric-train_val=kubric4hv_I4_5e-4_A_debug_2023.04.02_15.06.46 --dataset_type tapvid --dataset_location data/tapvid_davis/tapvid_davis.pkl --subset davis --query_mode first
python -m evaluate --modeltype pips --checkpoint_path logs/my_pips/checkpoints/1_8_768_72_tapvid-chunked_train=kubric-train_val=kubric4hv_I4_5e-4_A_debug_2023.04.02_15.06.46 --dataset_type tapvid --dataset_location none --subset kubric --query_mode first  
python -m evaluate --modeltype pips --checkpoint_path logs/my_pips/checkpoints/1_8_768_72_tapvid-chunked_train=kubric-train_val=kubric4hv_I4_5e-4_A_debug_2023.04.02_15.06.46 --dataset_type tapvid --dataset_location data/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl --subset rgb_stacking --query_mode first  
# 1_8_None_pips_72_tapvid_davis_first_2023.04.03_10.40.43
# 1_8_None_pips_72_tapvid_kubric_first_2023.04.03_10.41.00
# 1_8_None_pips_72_tapvid_rgb_stacking_first_2023.04.03_10.41.17
python -m pips_utils.figures --mostly_visible_threshold 4 --results_path_list \
  logs/1_8_None_pips_72_tapvid_davis_first_2023.04.03_10.40.43/results_df.csv \
  logs/1_8_None_pips_72_tapvid_kubric_first_2023.04.03_10.41.00/results_df.csv \
  logs/1_8_None_pips_72_tapvid_rgb_stacking_first_2023.04.03_10.41.17/results_df.csv
# TABLE: 'table3-pck-metrics'
# |                           |   ('0__pips_tapvid_davis', 'DAVIS', 'pips-8') |   ('1__pips_tapvid_kubric', 'KUBRIC', 'pips-8') |   ('2__pips_tapvid_rgb-stacking', 'RGB-STACKING', 'pips-8') |
# |:--------------------------|----------------------------------------------:|------------------------------------------------:|------------------------------------------------------------:|
# | ade                       |                                   69.9408     |                                      9.72678    |                                                 16.9478     |
# | ade_visible               |                                   17.4707     |                                      5.3563     |                                                 12.2655     |
# | ade_occluded              |                                  181.417      |                                     23.6842     |                                                 29.3814     |
# | ade_visible_chain         |                                    8.79814    |                                      4.15294    |                                                  7.28432    |
# | jaccard_1                 |                                    6.70658    |                                     24.8443     |                                                  4.26348    |
# | jaccard_2                 |                                   15.2856     |                                     48.5795     |                                                 10.7812     |
# | jaccard_4                 |                                   28.7727     |                                     65.5624     |                                                 24.0301     |
# | jaccard_8                 |                                   41.8762     |                                     75.0178     |                                                 42.8692     |
# | jaccard_16                |                                   50.8359     |                                     79.9472     |                                                 59.8077     |
# | pts_within_0.01           |                                    0.00811966 |                                      0.00592115 |                                                  0.00304862 |
# | pts_within_0.1            |                                    0.481478   |                                      0.542375   |                                                  0.348345   |
# | pts_within_0.5            |                                    5.84539    |                                      9.63866    |                                                  3.92568    |
# | pts_within_1              |                                   13.4866     |                                     34.1527     |                                                  9.32965    |
# | pts_within_2              |                                   27.2023     |                                     59.1781     |                                                 20.345      |
# | pts_within_4              |                                   45.3816     |                                     76.1846     |                                                 37.942      |
# | pts_within_8              |                                   61.0989     |                                     86.2884     |                                                 59.542      |
# | pts_within_16             |                                   72.8692     |                                     92.3354     |                                                 77.3803     |
# | average_jaccard           |                                   28.6954     |                                     58.7902     |                                                 28.3503     |
# | average_pts_within_thresh |                                   44.0077     |                                     69.6278     |                                                 40.9078     |
# | occlusion_accuracy        |                                   75.7947     |                                     87.9668     |                                                 82.0991     |

python -m evaluate --modeltype pips --checkpoint_path logs/my_pips/checkpoints/1_8_768_72_tapvid-chunked_train=kubric-train_val=kubric4hv_I4_5e-3_A_debug_2023.04.02_15.06.47 --dataset_type tapvid --dataset_location data/tapvid_davis/tapvid_davis.pkl --subset davis --query_mode first
python -m evaluate --modeltype pips --checkpoint_path logs/my_pips/checkpoints/1_8_768_72_tapvid-chunked_train=kubric-train_val=kubric4hv_I4_5e-3_A_debug_2023.04.02_15.06.47 --dataset_type tapvid --dataset_location none --subset kubric --query_mode first  
python -m evaluate --modeltype pips --checkpoint_path logs/my_pips/checkpoints/1_8_768_72_tapvid-chunked_train=kubric-train_val=kubric4hv_I4_5e-3_A_debug_2023.04.02_15.06.47 --dataset_type tapvid --dataset_location data/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl --subset rgb_stacking --query_mode first  
# 1_8_None_pips_72_tapvid_davis_first_2023.04.03_10.42.33
# 1_8_None_pips_72_tapvid_kubric_first_2023.04.03_10.42.40
# 1_8_None_pips_72_tapvid_rgb_stacking_first_2023.04.03_10.43.07
python -m pips_utils.figures --mostly_visible_threshold 4 --results_path_list \
  logs/1_8_None_pips_72_tapvid_davis_first_2023.04.03_10.42.33/results_df.csv \
  logs/1_8_None_pips_72_tapvid_kubric_first_2023.04.03_10.42.40/results_df.csv \
  logs/1_8_None_pips_72_tapvid_rgb_stacking_first_2023.04.03_10.43.07/results_df.csv
# TABLE: 'table3-pck-metrics'
# |                           |   ('0__pips_tapvid_davis', 'DAVIS', 'pips-8') |   ('1__pips_tapvid_kubric', 'KUBRIC', 'pips-8') |   ('2__pips_tapvid_rgb-stacking', 'RGB-STACKING', 'pips-8') |
# |:--------------------------|----------------------------------------------:|------------------------------------------------:|------------------------------------------------------------:|
# | ade                       |                                   69.6829     |                                     10.1157     |                                                 20.1573     |
# | ade_visible               |                                   18.466      |                                      6.06368    |                                                 15.0504     |
# | ade_occluded              |                                  178.522      |                                     23.9577     |                                                 31.3351     |
# | ade_visible_chain         |                                    9.95295    |                                      4.81975    |                                                  9.6458     |
# | jaccard_1                 |                                    5.21543    |                                     17.4833     |                                                  3.52874    |
# | jaccard_2                 |                                   12.5933     |                                     41.1527     |                                                  8.61705    |
# | jaccard_4                 |                                   26.0252     |                                     60.1279     |                                                 19.7403     |
# | jaccard_8                 |                                   38.7457     |                                     71.5531     |                                                 34.8082     |
# | jaccard_16                |                                   48.1484     |                                     77.5899     |                                                 51.3512     |
# | pts_within_0.01           |                                    0.00533333 |                                      0.00300215 |                                                  0.00131099 |
# | pts_within_0.1            |                                    0.243027   |                                      0.309458   |                                                  0.203457   |
# | pts_within_0.5            |                                    4.18291    |                                      6.157      |                                                  3.35111    |
# | pts_within_1              |                                   10.4648     |                                     25.6776     |                                                  8.15465    |
# | pts_within_2              |                                   22.5881     |                                     51.8481     |                                                 17.1347     |
# | pts_within_4              |                                   40.5811     |                                     71.2277     |                                                 32.8489     |
# | pts_within_8              |                                   57.1771     |                                     83.6013     |                                                 51.6662     |
# | pts_within_16             |                                   70.6294     |                                     91.1277     |                                                 70.0021     |
# | average_jaccard           |                                   26.1456     |                                     53.5814     |                                                 23.6091     |
# | average_pts_within_thresh |                                   40.2881     |                                     64.6965     |                                                 35.9613     |
# | occlusion_accuracy        |                                   73.0889     |                                     86.4645     |                                                 82.2685     |
```

Finetuning the PIPS checkpoint for three different values of the learning rate:
```bash
python train.py --dataset_type tapvid-chunked --dataset_location data/tapvid_kubric --subset_train kubric-train --subset_valid kubric --init_dir reference_model --load_optimizer True --load_step True --max_iters 250000 --B=1 --lr 5e-4
python train.py --dataset_type tapvid-chunked --dataset_location data/tapvid_kubric --subset_train kubric-train --subset_valid kubric --init_dir reference_model --load_optimizer True --max_iters 50000 --B=1 --lr 5e-5
python train.py --dataset_type tapvid-chunked --dataset_location data/tapvid_kubric --subset_train kubric-train --subset_valid kubric --init_dir reference_model --load_optimizer True --max_iters 50000 --B=1 --lr 5e-6
# logs/my_pips/checkpoints/1_8_768_72_tapvid-chunked_train=kubric-train_val=kubric4hv_I4_5e-4_A_debug_2023.04.02_15.07.18
# logs/my_pips/checkpoints/1_8_768_72_tapvid-chunked_train=kubric-train_val=kubric4hv_I4_5e-5_A_debug_2023.04.02_15.19.05
# logs/my_pips/checkpoints/1_8_768_72_tapvid-chunked_train=kubric-train_val=kubric4hv_I4_5e-6_A_debug_2023.04.02_15.07.17

python -m evaluate --modeltype pips --checkpoint_path logs/my_pips/checkpoints/1_8_768_72_tapvid-chunked_train=kubric-train_val=kubric4hv_I4_5e-4_A_debug_2023.04.02_15.07.18 --dataset_type tapvid --dataset_location data/tapvid_davis/tapvid_davis.pkl --subset davis --query_mode first   --log_freq 1
python -m evaluate --modeltype pips --checkpoint_path logs/my_pips/checkpoints/1_8_768_72_tapvid-chunked_train=kubric-train_val=kubric4hv_I4_5e-4_A_debug_2023.04.02_15.07.18 --dataset_type tapvid --dataset_location none --subset kubric --query_mode first  
python -m evaluate --modeltype pips --checkpoint_path logs/my_pips/checkpoints/1_8_768_72_tapvid-chunked_train=kubric-train_val=kubric4hv_I4_5e-4_A_debug_2023.04.02_15.07.18 --dataset_type tapvid --dataset_location data/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl --subset rgb_stacking --query_mode first  
python -m evaluate --modeltype pips --checkpoint_path logs/my_pips/checkpoints/1_8_768_72_tapvid-chunked_train=kubric-train_val=kubric4hv_I4_5e-4_A_debug_2023.04.02_15.07.18 --dataset_type tapvid --dataset_location data/tapvid_kinetics --subset kinetics --query_mode first  
# logs/1_8_None_pips_72_tapvid_davis_first_2023.04.03_00.39.13/results_df.csv
# logs/1_8_None_pips_72_tapvid_kubric_first_2023.04.03_00.39.16/results_df.csv
# logs/1_8_None_pips_72_tapvid_rgb_stacking_first_2023.04.03_00.40.05/results_df.csv
# logs/1_8_None_pips_72_tapvid_kinetics_first_2023.04.03_00.44.43/results_df.csv
python -m pips_utils.figures --mostly_visible_threshold 4 --results_path_list \
  logs/1_8_None_pips_72_tapvid_davis_first_2023.04.03_00.39.13/results_df.csv \
  logs/1_8_None_pips_72_tapvid_kubric_first_2023.04.03_00.39.16/results_df.csv \
  logs/1_8_None_pips_72_tapvid_rgb_stacking_first_2023.04.03_00.40.05/results_df.csv
  TABLE: 'table3-pck-metrics'
# |                           |   ('0__pips_tapvid_davis', 'DAVIS', 'pips-8') |   ('1__pips_tapvid_kubric', 'KUBRIC', 'pips-8') |   ('2__pips_tapvid_rgb-stacking', 'RGB-STACKING', 'pips-8') |
# |:--------------------------|----------------------------------------------:|------------------------------------------------:|------------------------------------------------------------:|
# | ade                       |                                    66.8165    |                                     10.4113     |                                                19.821       |
# | ade_visible               |                                    15.9035    |                                      4.67213    |                                                15.5134      |
# | ade_occluded              |                                   175.739     |                                     26.4425     |                                                29.4764      |
# | ade_visible_chain         |                                     7.74776   |                                      3.58124    |                                                10.3858      |
# | jaccard_1                 |                                     5.29771   |                                     13.7317     |                                                 2.04148     |
# | jaccard_2                 |                                    13.1285    |                                     41.5484     |                                                 5.60477     |
# | jaccard_4                 |                                    26.7023    |                                     64.7561     |                                                13.8445      |
# | jaccard_8                 |                                    41.983     |                                     76.1274     |                                                30.2123      |
# | jaccard_16                |                                    52.1758    |                                     81.0443     |                                                49.5183      |
# | pts_within_0.01           |                                     0.0047619 |                                      0.00155196 |                                                 0.000267738 |
# | pts_within_0.1            |                                     0.114513  |                                      0.199799   |                                                 0.114452    |
# | pts_within_0.5            |                                     3.98179   |                                      5.14062    |                                                 1.86299     |
# | pts_within_1              |                                    10.7309    |                                     21.3386     |                                                 4.96504     |
# | pts_within_2              |                                    24.1047    |                                     53.458      |                                                12.0801      |
# | pts_within_4              |                                    43.4856    |                                     75.9279     |                                                25.4458      |
# | pts_within_8              |                                    62.8233    |                                     87.9085     |                                                47.5282      |
# | pts_within_16             |                                    77.7082    |                                     94.1404     |                                                69.5235      |
# | average_jaccard           |                                    27.8575    |                                     55.4416     |                                                20.2443      |
# | average_pts_within_thresh |                                    43.7705    |                                     66.5547     |                                                31.9085      |
# | occlusion_accuracy        |                                    71.6156    |                                     87.8709     |                                                80.1063      |


python -m evaluate --modeltype pips --checkpoint_path logs/my_pips/checkpoints/1_8_768_72_tapvid-chunked_train=kubric-train_val=kubric4hv_I4_5e-5_A_debug_2023.04.02_15.19.05 --dataset_type tapvid --dataset_location data/tapvid_davis/tapvid_davis.pkl --subset davis --query_mode first   --log_freq 1
python -m evaluate --modeltype pips --checkpoint_path logs/my_pips/checkpoints/1_8_768_72_tapvid-chunked_train=kubric-train_val=kubric4hv_I4_5e-5_A_debug_2023.04.02_15.19.05 --dataset_type tapvid --dataset_location none --subset kubric --query_mode first  
python -m evaluate --modeltype pips --checkpoint_path logs/my_pips/checkpoints/1_8_768_72_tapvid-chunked_train=kubric-train_val=kubric4hv_I4_5e-5_A_debug_2023.04.02_15.19.05 --dataset_type tapvid --dataset_location data/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl --subset rgb_stacking --query_mode first  
python -m evaluate --modeltype pips --checkpoint_path logs/my_pips/checkpoints/1_8_768_72_tapvid-chunked_train=kubric-train_val=kubric4hv_I4_5e-5_A_debug_2023.04.02_15.19.05 --dataset_type tapvid --dataset_location data/tapvid_kinetics --subset kinetics --query_mode first  
# 1_8_None_pips_72_tapvid_davis_first_2023.04.03_00.44.54
# 1_8_None_pips_72_tapvid_kubric_first_2023.04.03_00.45.52
# 1_8_None_pips_72_tapvid_rgb_stacking_first_2023.04.03_00.47.56
# 1_8_None_pips_72_tapvid_kinetics_first_2023.04.03_00.48.00
python -m pips_utils.figures --mostly_visible_threshold 4 --results_path_list \
  logs/1_8_None_pips_72_tapvid_davis_first_2023.04.03_00.44.54/results_df.csv \
  logs/1_8_None_pips_72_tapvid_kubric_first_2023.04.03_00.45.52/results_df.csv \
  logs/1_8_None_pips_72_tapvid_rgb_stacking_first_2023.04.03_00.47.56/results_df.csv
# TABLE: 'table3-pck-metrics'
# |                           |   ('0__pips_tapvid_davis', 'DAVIS', 'pips-8') |   ('1__pips_tapvid_kubric', 'KUBRIC', 'pips-8') |   ('2__pips_tapvid_rgb-stacking', 'RGB-STACKING', 'pips-8') |
# |:--------------------------|----------------------------------------------:|------------------------------------------------:|------------------------------------------------------------:|
# | ade                       |                                   64.0511     |                                     12.5006     |                                                  18.0782    |
# | ade_visible               |                                   11.3549     |                                      3.6459     |                                                  14.3461    |
# | ade_occluded              |                                  177.164      |                                     26.2897     |                                                  24.7816    |
# | ade_visible_chain         |                                    6.07549    |                                      2.86753    |                                                  10.253     |
# | jaccard_1                 |                                    5.13517    |                                     17.7128     |                                                   2.23988   |
# | jaccard_2                 |                                   13.6006     |                                     44.7915     |                                                   5.61954   |
# | jaccard_4                 |                                   30.8308     |                                     69.9393     |                                                  13.2043    |
# | jaccard_8                 |                                   50.3454     |                                     82.0667     |                                                  29.9091    |
# | jaccard_16                |                                   63.2797     |                                     86.255      |                                                  50.651     |
# | pts_within_0.01           |                                    0.00246914 |                                      0.00365241 |                                                   0.0359524 |
# | pts_within_0.1            |                                    0.200504   |                                      0.336546   |                                                   0.191372  |
# | pts_within_0.5            |                                    3.65793    |                                      7.55928    |                                                   2.40987   |
# | pts_within_1              |                                   10.4971     |                                     26.834      |                                                   5.96326   |
# | pts_within_2              |                                   25.1854     |                                     56.8913     |                                                  12.5911    |
# | pts_within_4              |                                   48.5736     |                                     80.4795     |                                                  24.8339    |
# | pts_within_8              |                                   71.0335     |                                     91.7381     |                                                  46.347     |
# | pts_within_16             |                                   84.9991     |                                     96.3597     |                                                  69.3215    |
# | average_jaccard           |                                   32.6383     |                                     60.1531     |                                                  20.3248    |
# | average_pts_within_thresh |                                   48.0577     |                                     70.4605     |                                                  31.8114    |
# | occlusion_accuracy        |                                   79.2124     |                                     91.1377     |                                                  83.7102    |

python -m evaluate --modeltype pips --checkpoint_path logs/my_pips/checkpoints/1_8_768_72_tapvid-chunked_train=kubric-train_val=kubric4hv_I4_5e-6_A_debug_2023.04.02_15.07.17 --dataset_type tapvid --dataset_location data/tapvid_davis/tapvid_davis.pkl --subset davis --query_mode first   --log_freq 1
python -m evaluate --modeltype pips --checkpoint_path logs/my_pips/checkpoints/1_8_768_72_tapvid-chunked_train=kubric-train_val=kubric4hv_I4_5e-6_A_debug_2023.04.02_15.07.17 --dataset_type tapvid --dataset_location none --subset kubric --query_mode first  
python -m evaluate --modeltype pips --checkpoint_path logs/my_pips/checkpoints/1_8_768_72_tapvid-chunked_train=kubric-train_val=kubric4hv_I4_5e-6_A_debug_2023.04.02_15.07.17 --dataset_type tapvid --dataset_location data/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl --subset rgb_stacking --query_mode first  
python -m evaluate --modeltype pips --checkpoint_path logs/my_pips/checkpoints/1_8_768_72_tapvid-chunked_train=kubric-train_val=kubric4hv_I4_5e-6_A_debug_2023.04.02_15.07.17 --dataset_type tapvid --dataset_location data/tapvid_kinetics --subset kinetics --query_mode first  
# 1_8_None_pips_72_tapvid_davis_first_2023.04.03_00.49.49
# 1_8_None_pips_72_tapvid_kubric_first_2023.04.03_00.49.53
# 1_8_None_pips_72_tapvid_rgb_stacking_first_2023.04.03_00.49.56
# 1_8_None_pips_72_tapvid_kinetics_first_2023.04.03_00.50.00
python -m pips_utils.figures --mostly_visible_threshold 4 --results_path_list \
  logs/1_8_None_pips_72_tapvid_davis_first_2023.04.03_00.49.49/results_df.csv \
  logs/1_8_None_pips_72_tapvid_kubric_first_2023.04.03_00.49.53/results_df.csv \
  logs/1_8_None_pips_72_tapvid_rgb_stacking_first_2023.04.03_00.49.56/results_df.csv
# |                           |   ('0__pips_tapvid_davis', 'DAVIS', 'pips-8') |   ('1__pips_tapvid_kubric', 'KUBRIC', 'pips-8') |   ('2__pips_tapvid_rgb-stacking', 'RGB-STACKING', 'pips-8') |
# |:--------------------------|----------------------------------------------:|------------------------------------------------:|------------------------------------------------------------:|
# | ade                       |                                     64.0612   |                                      7.32331    |                                                 15.9427     |
# | ade_visible               |                                      9.83197  |                                      3.46685    |                                                 11.6705     |
# | ade_occluded              |                                    180.667    |                                     18.0969     |                                                 24.5902     |
# | ade_visible_chain         |                                      5.15454  |                                      2.68642    |                                                  7.94242    |
# | jaccard_1                 |                                      6.80641  |                                     19.0881     |                                                  2.96895    |
# | jaccard_2                 |                                     17.2271   |                                     48.7124     |                                                  7.45769    |
# | jaccard_4                 |                                     36.3449   |                                     72.6142     |                                                 18.0976     |
# | jaccard_8                 |                                     58.0345   |                                     82.4576     |                                                 38.6585     |
# | jaccard_16                |                                     67.2292   |                                     86.0041     |                                                 59.3845     |
# | pts_within_0.01           |                                      0        |                                      0.00397133 |                                                  0.00189043 |
# | pts_within_0.1            |                                      0.164743 |                                      0.334542   |                                                  0.153338   |
# | pts_within_0.5            |                                      4.57935  |                                      8.01407    |                                                  2.87367    |
# | pts_within_1              |                                     13.4753   |                                     28.6128     |                                                  7.39973    |
# | pts_within_2              |                                     30.0282   |                                     60.6205     |                                                 15.9429     |
# | pts_within_4              |                                     54.3677   |                                     82.4288     |                                                 31.7942     |
# | pts_within_8              |                                     77.4692   |                                     92.0969     |                                                 56.2518     |
# | pts_within_16             |                                     88.1974   |                                     96.4016     |                                                 77.7463     |
# | average_jaccard           |                                     37.1284   |                                     61.7753     |                                                 25.3135     |
# | average_pts_within_thresh |                                     52.7076   |                                     72.0321     |                                                 37.8269     |
# | occlusion_accuracy        |                                     80.2266   |                                     90.8563     |                                                 83.2159     |
```

Did the performance on tapvid_chunked kubric improve as it should since we trained on it?
```bash
ssh i58
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
export CUDA_VISIBLE_DEVICES=0
source pips.sh

# Sanity check
python -m evaluate --modeltype pips --dataset_type tapvid-chunked --dataset_location none --subset kubric --query_mode first
# logs/1_8_None_pips_72_tapvid-chunked_kubric_2023.04.03_09.42.49/figures

python -m evaluate --modeltype pips --dataset_type tapvid-chunked --dataset_location none --subset kubric --query_mode first --checkpoint_path logs/my_pips/checkpoints/1_8_768_72_tapvid-chunked_train=kubric-train_val=kubric4hv_I4_5e-4_A_debug_2023.04.02_15.07.18
python -m evaluate --modeltype pips --dataset_type tapvid-chunked --dataset_location none --subset kubric --query_mode first --checkpoint_path logs/my_pips/checkpoints/1_8_768_72_tapvid-chunked_train=kubric-train_val=kubric4hv_I4_5e-5_A_debug_2023.04.02_15.19.05
python -m evaluate --modeltype pips --dataset_type tapvid-chunked --dataset_location none --subset kubric --query_mode first --checkpoint_path logs/my_pips/checkpoints/1_8_768_72_tapvid-chunked_train=kubric-train_val=kubric4hv_I4_5e-6_A_debug_2023.04.02_15.07.17
# logs/1_8_None_pips_72_tapvid-chunked_kubric_2023.04.03_09.42.24/figures
# logs/1_8_None_pips_72_tapvid-chunked_kubric_2023.04.03_09.58.54/figures
# logs/1_8_None_pips_72_tapvid-chunked_kubric_2023.04.03_09.58.31/figures

python -m evaluate --modeltype pips --dataset_type tapvid-chunked --dataset_location none --subset kubric --query_mode first --checkpoint_path logs/my_pips/checkpoints/1_8_768_72_tapvid-chunked_train=kubric-train_val=kubric4hv_I4_5e-5_A_debug_2023.04.02_15.07.04
# logs/1_8_None_pips_72_tapvid-chunked_kubric_2023.04.03_09.58.37/figures
# 
# 

python -m pips_utils.figures --mostly_visible_threshold 4 --results_path_list \
  logs/1_8_None_pips_72_tapvid-chunked_kubric_2023.04.03_09.42.49/results_df.csv \
  logs/1_8_None_pips_72_tapvid-chunked_kubric_2023.04.03_09.42.24/results_df.csv \
  logs/1_8_None_pips_72_tapvid-chunked_kubric_2023.04.03_09.58.54/results_df.csv \
  logs/1_8_None_pips_72_tapvid-chunked_kubric_2023.04.03_09.58.31/results_df.csv \
  logs/1_8_None_pips_72_tapvid-chunked_kubric_2023.04.03_09.58.37/results_df.csv


```

## Visualize trajectories per dataset per point category (standard, hard, easy)

```bash
# export CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt

python -m evaluate --modeltype pips --dataset_type tapvid --dataset_location data/tapvid_davis/tapvid_davis.pkl --subset davis --query_mode first --log_freq 1 --max_iter 30 --wandb_project evaluation-tapvid-trajectories
python -m evaluate --modeltype pips --dataset_type tapvid --dataset_location none --subset kubric --query_mode first --log_freq 1 --max_iter 30 --wandb_project evaluation-tapvid-trajectories
python -m evaluate --modeltype pips --dataset_type tapvid --dataset_location data/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl --subset rgb_stacking --query_mode first --log_freq 1 --max_iter 30 --wandb_project evaluation-tapvid-trajectories
python -m evaluate --modeltype pips --dataset_type tapvid --dataset_location data/tapvid_kinetics --subset kinetics --query_mode first --log_freq 1 --max_iter 30 --wandb_project evaluation-tapvid-trajectories

python -m evaluate --modeltype raft --dataset_type tapvid --dataset_location data/tapvid_davis/tapvid_davis.pkl --subset davis --query_mode first --log_freq 1 --max_iter 30 --wandb_project evaluation-tapvid-trajectories
python -m evaluate --modeltype raft --dataset_type tapvid --dataset_location none --subset kubric --query_mode first --log_freq 1 --max_iter 30 --wandb_project evaluation-tapvid-trajectories
python -m evaluate --modeltype raft --dataset_type tapvid --dataset_location data/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl --subset rgb_stacking --query_mode first --log_freq 1 --max_iter 30 --wandb_project evaluation-tapvid-trajectories
python -m evaluate --modeltype raft --dataset_type tapvid --dataset_location data/tapvid_kinetics --subset kinetics --query_mode first --log_freq 1 --max_iter 30 --wandb_project evaluation-tapvid-trajectories
```