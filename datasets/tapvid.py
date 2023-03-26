import torch


class TAPVidIterator(torch.utils.data.Dataset):
    def __init__(self, dataset_path, tapvid_subset_name, query_mode):
        self.dataset_path = dataset_path
        self.tapvid_subset_name = tapvid_subset_name
        self.query_mode = query_mode

    def _generate_datapoints(self):
        return list(iter(self))

    def __iter__(self):
        if self.tapvid_subset_name == "kinetics":
            from datasets.tapvid_evaluation_datasets import create_kinetics_dataset
            dataset_element_iterator = create_kinetics_dataset(self.dataset_path, self.query_mode)
        elif self.tapvid_subset_name == "davis":
            from datasets.tapvid_evaluation_datasets import create_davis_dataset
            dataset_element_iterator = create_davis_dataset(self.dataset_path, self.query_mode)
        elif self.tapvid_subset_name == "kubric":
            from datasets.tapvid_evaluation_datasets import create_kubric_eval_dataset
            dataset_element_iterator = create_kubric_eval_dataset(mode="")  # TODO What mode to use for kubric?
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

        assert batch_size == 1
        assert rgbs.shape == (batch_size, n_frames, channels, height, width)
        assert query_points.shape == (batch_size, n_points, 3)
        assert trajectories.shape == (batch_size, n_frames, n_points, 2)
        assert visibilities.shape == (batch_size, n_frames, n_points)

        # Rescale from [-1,1] to [0,255]
        rgbs = (rgbs + 1) * 255 / 2

        # Convert query points from (t, y, x) to (t, x, y)
        query_points = query_points[:, :, [0, 2, 1]]

        for point_idx, query_point in enumerate(query_points[0]):
            if not torch.allclose(
                    trajectories[0, int(query_point[0]), point_idx].float(),
                    query_point[1:].float(),
                    atol=0.51,
            ):
                print(
                    f"WARNING: Query point does not match trajectory: {trajectories[0, int(query_point[0]), point_idx].float()} {query_point[1:].float()}")

        return {
            "rgbs": rgbs,
            "query_points": query_points,
            "trajectories": trajectories,
            "visibilities": visibilities,
        }
