import warnings

import torch


class TAPVidIterator(torch.utils.data.Dataset):
    """
    An iterator that loads a TAP-Vid dataset and yields its elements.
    The elements consist of videos of arbitrary length.
    """

    def __init__(self, dataset_path, tapvid_subset_name, query_mode):
        self.dataset_path = dataset_path
        self.tapvid_subset_name = tapvid_subset_name
        self.query_mode = query_mode

    def __iter__(self):
        # Do not let TF use GPUs, assuming that TF is not used elsewhere.
        import tensorflow as tf
        cpus = tf.config.list_physical_devices('CPU')
        assert len(cpus) > 0
        tf.config.set_visible_devices(cpus)

        if self.tapvid_subset_name == "kinetics":
            from datasets.tapvid_evaluation_datasets import create_kinetics_dataset
            dataset_element_iterator = create_kinetics_dataset(self.dataset_path, self.query_mode)
        elif self.tapvid_subset_name == "davis":
            from datasets.tapvid_evaluation_datasets import create_davis_dataset
            dataset_element_iterator = create_davis_dataset(self.dataset_path, self.query_mode)
        elif self.tapvid_subset_name == "kubric":
            from datasets.tapvid_evaluation_datasets import create_kubric_eval_dataset
            dataset_element_iterator = create_kubric_eval_dataset(mode="")  # TODO What mode to use for kubric?
        elif self.tapvid_subset_name == "kubric-train":
            from datasets.tapvid_evaluation_datasets import create_kubric_eval_train_dataset
            dataset_element_iterator = create_kubric_eval_train_dataset(mode="", max_dataset_size=None)
        elif self.tapvid_subset_name == "rgb_stacking":
            from datasets.tapvid_evaluation_datasets import create_rgb_stacking_dataset
            dataset_element_iterator = create_rgb_stacking_dataset(self.dataset_path, self.query_mode)
        elif self.tapvid_subset_name == "jhmdb":
            from datasets.tapvid_evaluation_datasets import create_jhmdb_dataset
            dataset_element_iterator = create_jhmdb_dataset(self.dataset_path)
        else:
            raise ValueError(f"TAP-Vid subset name `{self.tapvid_subset_name}` is not supported.")

        for i, dataset_element in enumerate(dataset_element_iterator):
            assert len(dataset_element.values()) == 1
            dataset_element = list(dataset_element.values())[0]
            yield TAPVidIterator.preprocess_dataset_element(dataset_element)

    @staticmethod
    def preprocess_dataset_element(dataset_element):
        rgbs = torch.from_numpy(dataset_element['video']).permute(0, 1, 4, 2, 3)
        query_points = torch.from_numpy(dataset_element['query_points'])
        trajectories = torch.from_numpy(dataset_element['target_points']).permute(0, 2, 1, 3)
        visibilities = ~torch.from_numpy(dataset_element['occluded']).permute(0, 2, 1)

        batch_size, n_frames, channels, height, width = rgbs.shape
        n_points = query_points.shape[1]

        # Rescale from [-1,1] to [0,255]
        rgbs = (rgbs + 1) * 255 / 2

        # Convert query points from (t, y, x) to (t, x, y)
        query_points = query_points[:, :, [0, 2, 1]]

        # Ad hoc fix for Kubric reporting invisible query points when close to the crop boundary, e.g., x=110, y=-1e-5
        for point_idx in range(n_points):
            query_point = query_points[0, point_idx]
            query_visible = visibilities[0, query_point[0].long(), point_idx]
            if query_visible:
                continue

            x, y = query_point[1:]
            x_at_boundary = min(abs(x - 0), abs(x - (width - 1))) < 1e-3
            y_at_boundary = min(abs(y - 0), abs(y - (height - 1))) < 1e-3
            x_inside_window = 0 <= x <= width - 1
            y_inside_window = 0 <= y <= height - 1

            if x_at_boundary and y_inside_window or x_inside_window and y_at_boundary or x_at_boundary and y_at_boundary:
                visibilities[0, query_point[0].long(), point_idx] = 1

        # Check dimensions are correct
        assert batch_size == 1
        assert rgbs.shape == (batch_size, n_frames, channels, height, width)
        assert query_points.shape == (batch_size, n_points, 3)
        assert trajectories.shape == (batch_size, n_frames, n_points, 2)
        assert visibilities.shape == (batch_size, n_frames, n_points)

        # Check that query points are visible
        assert torch.all(visibilities[0, query_points[0, :, 0].long(), torch.arange(n_points)] == 1), \
            "Query points must be visible"

        # Check that query points are correct
        assert torch.allclose(
            query_points[0, :, 1:].float(),
            trajectories[0, query_points[0, :, 0].long(), torch.arange(n_points)].float(),
            atol=1.0,
        )

        return {
            "rgbs": rgbs,
            "query_points": query_points,
            "trajectories": trajectories,
            "visibilities": visibilities,
        }


class TAPVidChunkedDataset(torch.utils.data.Dataset):
    """
    A dataset that loads a TAP-Vid dataset and splits it into chunks of a given chunk length.
    PIPS is trained on these chunks, but the evaluation is done on the full sequences (see TAPVidIterator).
    The whole dataset is loaded to memory at once and optionally cached to disk.
    """

    def __init__(self, dataset_path, tapvid_subset_name, chunk_length=8, chunking_stride=2):
        self.iterator = TAPVidIterator(dataset_path, tapvid_subset_name, query_mode="first")
        self.dataset_path = dataset_path
        self.tapvid_subset_name = tapvid_subset_name
        self.chunk_length = chunk_length
        self.chunking_stride = chunking_stride

    def __iter__(self):
        for i, dataset_element in enumerate(self.iterator):
            print(f"Yield datapoint {i}.")
            yield from self._dataset_element_to_sequences(dataset_element)

    def _dataset_element_to_sequences(self, dataset_element):
        rgbs = dataset_element["rgbs"]
        _ = dataset_element["query_points"]
        trajectories = dataset_element["trajectories"]
        visibilities = dataset_element["visibilities"]

        batch_size, n_frames, channels, height, width = rgbs.shape
        assert batch_size == 1

        for sequence_start in range(0, n_frames, self.chunking_stride):
            sequence_end = sequence_start + self.chunk_length
            sequence_range = slice(sequence_start, sequence_end)
            if sequence_end > n_frames:
                break

            first_timestep_visible = visibilities[:, sequence_range, :][0, 0, :]
            if not first_timestep_visible.any():
                warnings.warn(f"Skipping datapoint subsequence with sequence_start={sequence_start}: "
                              f"No first timestep was visible in trajectories.")
                continue

            rgbs_datapoint = rgbs[:, sequence_range, :, :, :]
            trajectories_datapoint = trajectories[:, sequence_range, first_timestep_visible, :]
            visibilities_datapoint = visibilities[:, sequence_range, first_timestep_visible]

            # Query points are the first timestep of the trajectories
            query_point_xy = trajectories_datapoint[:, 0, :, :]
            query_point_timestep = torch.zeros((query_point_xy.shape[0], query_point_xy.shape[1], 1))
            query_points_datapoint = torch.cat([query_point_timestep, query_point_xy], dim=2)

            datapoint = {
                "rgbs": rgbs_datapoint,
                "query_points": query_points_datapoint,
                "trajectories": trajectories_datapoint,
                "visibilities": visibilities_datapoint,
            }
            yield datapoint
