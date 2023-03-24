import os
import pickle
import warnings

import numpy as np
import torch


class TAPVidDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, tapvid_subset_name, sequence_length, cache=True):
        self.dataset_path = dataset_path
        self.tapvid_subset_name = tapvid_subset_name
        self.sequence_length = sequence_length

        if cache:
            self.datapoints = self._load_cached_datapoints()
        else:
            print(f"Generating datapoints.")
            self.datapoints = self._generate_datapoints()

    def _generate_datapoints(self):
        return list(iter(self))

    def _load_cached_datapoints(self):
        self.cached_datapoints_path = self.dataset_path
        while self.cached_datapoints_path.endswith("/"):
            self.cached_datapoints_path = self.cached_datapoints_path[:-1]
        self.cached_datapoints_path += f".cached"
        self.cached_datapoints_path += f".{self.sequence_length}"
        self.cached_datapoints_path += f".{self.tapvid_subset_name}"
        self.cached_datapoints_path += f".pkl"

        if os.path.isfile(self.cached_datapoints_path):
            print(f"Loading cached datapoints from:\n{os.path.abspath(self.cached_datapoints_path)}")
            with open(self.cached_datapoints_path, "rb") as f:
                datapoints = pickle.load(f)
        else:
            print(f"Cached datapoints not found at:\n{os.path.abspath(self.cached_datapoints_path)}")
            print(f"Generating datapoints.")
            datapoints = self._generate_datapoints()
            with open(self.cached_datapoints_path, "wb") as f:
                pickle.dump(datapoints, f)
            print(f"Datapoints cached to:\n{os.path.abspath(self.cached_datapoints_path)}")

        return datapoints

    def __iter__(self):
        if self.tapvid_subset_name == "kinetics":
            from datasets.tapvid_evaluation_datasets import create_kinetics_dataset
            dataset_element_iterator = create_kinetics_dataset(kinetics_path=self.dataset_path, query_mode='first')
        elif self.tapvid_subset_name == "davis":
            from datasets.tapvid_evaluation_datasets import create_davis_dataset
            dataset_element_iterator = create_davis_dataset(davis_points_path=self.dataset_path, query_mode='first')
        elif self.tapvid_subset_name == "kubric":
            from datasets.tapvid_evaluation_datasets import create_kubric_eval_dataset
            dataset_element_iterator = create_kubric_eval_dataset(mode="")  # TODO What mode to use for kubric?
        elif self.tapvid_subset_name == "rgb_stacking":
            from datasets.tapvid_evaluation_datasets import create_rgb_stacking_dataset
            dataset_element_iterator = create_rgb_stacking_dataset(self.dataset_path, query_mode="first")
        elif self.tapvid_subset_name == "jhmdb":
            from datasets.tapvid_evaluation_datasets import create_jhmdb_dataset
            dataset_element_iterator = create_jhmdb_dataset(jhmdb_path=self.dataset_path)
        else:
            raise ValueError(f"TAP-Vid subset name `{self.tapvid_subset_name}` is not supported.")

        for i, dataset_element in enumerate(dataset_element_iterator):
            assert len(dataset_element.values()) == 1
            dataset_element = list(dataset_element.values())[0]
            print(f"Yield datapoint {i}.")
            yield from self._dataset_element_to_sequences(dataset_element)

    def _dataset_element_to_sequences(self, dataset_element):
        video = torch.from_numpy(dataset_element['video'])
        query_points = torch.from_numpy(dataset_element['query_points'])
        target_points = torch.from_numpy(dataset_element['target_points'])
        occluded = torch.from_numpy(dataset_element['occluded'])

        batch_size, n_frames, height, width, channels = video.shape
        n_points = query_points.shape[1]

        assert batch_size == 1
        assert video.shape == (batch_size, n_frames, height, width, channels)
        assert query_points.shape == (batch_size, n_points, 3)
        assert target_points.shape == (batch_size, n_points, n_frames, 2)
        assert occluded.shape == (batch_size, n_points, n_frames)

        for sequence_start in range(0, n_frames // self.sequence_length * self.sequence_length, self.sequence_length):
            sequence_range = np.arange(sequence_start, sequence_start + self.sequence_length)
            rgbs = video[0, sequence_range, :, :, :].permute(0, 3, 1, 2)
            trajectories = target_points[0, :, sequence_range, :].permute(1, 0, 2)
            visibility = ~occluded[0, :, sequence_range].permute(1, 0)

            assert rgbs.shape == (self.sequence_length, channels, height, width)
            assert trajectories.shape == (self.sequence_length, n_points, 2)
            assert visibility.shape == (self.sequence_length, n_points)

            first_two_timesteps_visible = visibility[:2, :].sum(0) == 2
            if not first_two_timesteps_visible.any():
                warnings.warn(f"Frist two timesteps were not visible for all trajectories "
                              f"for sequence_start={sequence_start} (n_frames={n_frames}). "
                              f"Skipping datapoint.")
                continue

            datapoint = {
                "rgbs": rgbs,
                "trajectories": trajectories[:, first_two_timesteps_visible],
                "visibility": visibility[:, first_two_timesteps_visible],
            }

            yield datapoint

    def __getitem__(self, idx):
        return self.datapoints[idx]

    def __len__(self):
        return len(self.datapoints)
