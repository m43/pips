from torch.nn import functional as F
from torch.utils.data import DataLoader

from datasets.flyingthings import FlyingThingsDataset
from datasets.tapvid import TAPVidIterator, TAPVidChunkedDataset
from pips_utils.util import worker_seed_init_fn


class DataloaderFactory:
    @classmethod
    def get_dataloader(cls, name, dataset_location, subset, query_mode, pips_window, flt_use_augs, flt_crop_size,
                       n_points, batch_size, shuffle, dataloader_workers):
        if name == "flyingthings++":
            subset, dset = subset.split("-", maxsplit=1)  # e.g., all-train -> all, train
            dset = dset.upper()
            dataset = FlyingThingsDataset(
                dataset_location=dataset_location,
                dset=dset, subset=subset,
                use_augs=flt_use_augs,
                N=n_points, S=pips_window,
                crop_size=flt_crop_size,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=dataloader_workers,
                worker_init_fn=worker_seed_init_fn,
                drop_last=True,
            )

        elif name == "tapvid":
            dataloader = TAPVidIterator(dataset_location, subset, query_mode)

        elif name == "tapvid-chunked":
            dataloader = TAPVidChunkedDataset(dataset_location, subset, pips_window)

        elif name == "crohd":
            raise NotImplementedError()

        else:
            raise ValueError(f"Invalid dataset type given: `{name}`")

        return dataloader

    @classmethod
    def unpack_batch(cls, batch, dataset, modeltype, device):
        # TODO Refactor: Move to respective dataloaders
        if dataset == "flyingthings++":
            rgbs = batch['rgbs'].to(device).float()
            query_points = None  # TODO
            trajectories_gt = batch['trajs'].to(device).float()
            visibilities_gt = batch['visibles'].to(device).float()

        elif dataset == "crohd":
            rgbs = batch['rgbs'].to(device)
            query_points = None  # TODO
            trajectories_gt = batch['trajs_g'].to(device)
            visibilities_gt = batch['vis_g'].to(device)

            batch_size, n_frames, channels, height, width = rgbs.shape
            batch_size, S1, n_points, D = trajectories_gt.shape

            # TODO Remove the resizing for DINO if possible
            if modeltype == "dino":
                H_, W_ = 512, 768
            else:
                H_, W_ = 768, 1280
            sy = H_ / height
            sx = W_ / width

            rgbs_ = rgbs.reshape(batch_size * n_frames, channels, height, width)
            rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
            height, width = H_, W_
            rgbs = rgbs_.reshape(batch_size, n_frames, channels, height, width)
            trajectories_gt[:, :, :, 0] *= sx
            trajectories_gt[:, :, :, 1] *= sy

        elif dataset in ["tapvid", "tapvid-chunked"]:
            rgbs = batch["rgbs"].to(device)
            query_points = batch["query_points"].to(device)
            trajectories_gt = batch["trajectories"].to(device)
            visibilities_gt = batch["visibilities"].to(device)

        else:
            raise ValueError(f"Invalid dataset given: `{dataset}`")

        batch_size, n_frames, channels, height, width = rgbs.shape
        n_points = trajectories_gt.shape[2]

        assert rgbs.shape == (batch_size, n_frames, channels, height, width)
        assert query_points.shape == (batch_size, n_points, 3)
        assert trajectories_gt.shape == (batch_size, n_frames, n_points, 2)
        assert visibilities_gt.shape == (batch_size, n_frames, n_points)

        return rgbs, query_points, trajectories_gt, visibilities_gt
