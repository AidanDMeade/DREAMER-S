import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from operator import itemgetter
import numpy as np
import pandas as pd

try:
    from specio.io import SpecFileReader
    reader = SpecFileReader
    use_specio = True
    print("Spec reader: using SpecFileReader from specio.io")
except ImportError:
    from biospecml.processings.file_readers import read_mat
    reader = read_mat
    use_specio = False
    print("Spec reader: Using read_mat from biospecml")


labels_dict = {
    "CRC0076": 0,
    "CRC0344": 1,
    }

def read_metadata(
    metadata_path,
    filesdict: dict,
    label_col,
    id_col,
    ):

    # read metadata from the train/val split
    df_meta = pd.read_csv(metadata_path, index_col=0, delimiter="\t")
    print(f"Total labels in metadata: {len(df_meta)}")
    print(df_meta[label_col].value_counts())

    # filter main files dict to split files only
    filesdict_split = {
        k: v for k, v in filesdict.items() if k in df_meta[id_col].to_list()
    }
    print(f"Total file dict after filtered: {len(filesdict_split)}")

    # check if metadata tally with filtered files dict
    if len(filesdict_split) != len(df_meta):
        raise Exception("Metadata not tally to available files!")

    return df_meta, filesdict_split

class ChemImgDataset(Dataset):
    """
    A custom dataset for chemical image.
    Developed by @rafsanlab, April 2025, Ireland.

    Args:
        filesdict = {fileid:filepath}

    """

    def __init__(
        self,
        filesdict: dict,
        metadata_df: pd.DataFrame,
        id_col,
        label_col,
        labels_dict: dict = None,
        mean: np.ndarray = None,
        std: np.ndarray = None,
        return_datadict: bool = False,
        instrument: str = "spero", # or 'agilent',
    ):
        self.metadata_df = metadata_df
        self.id_col = id_col
        self.label_col = label_col
        self.labels_dict = labels_dict
        self.filesdict = filesdict
        self.ids = list(filesdict.keys())
        self.mean = mean
        self.std = std
        self.instrument = instrument
        self.return_datadict = return_datadict
        self._check_total_file()

    # __________________________________________________________________________

    def _check_total_file(self):
        if len(self.metadata_df) != len(self.filesdict):
            raise ValueError(
                "Total images in filesdict not equal to the provided metadata dataframe."
            )

    # __________________________________________________________________________

    def __len__(self):
        return len(self.filesdict)

    # __________________________________________________________________________

    def add_mean_std(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std = std

    # __________________________________________________________________________

    def _normalise(self, data):
        if self.mean is not None and self.std is not None:
            return (data - self.mean) / (self.std + 1e-6)
        return data

    # __________________________________________________________________________

    def __getitem__(self, idx):

        sample_id = self.ids[idx]
        sample_path = self.filesdict[sample_id]

        # get label from the metadata_df
        label = self.metadata_df.loc[
            self.metadata_df[self.id_col] == sample_id, self.label_col
        ].values[0]
        if self.labels_dict is not None:
            label = self.labels_dict[label]
        label = torch.tensor(label, dtype=torch.long)
        
        if use_specio:
            datadict = reader.read(
                filepath=sample_path,
                datatype="mat",
                spatial=True,
                strict_mode=False,
            )
            wn, s, sp, z, w, h = itemgetter("wn", "s", "sp", "z", "w", "h")(datadict)
            s = s.transpose()
        else:
            w, h, p, wn, sp = read_mat(sample_path, instrument=self.instrument)
            s = p

        data = torch.tensor(s, dtype=torch.float32)
        data = self._normalise(data)

        if self.return_datadict:
            return data, label, datadict
        else:
            return data, label

    # __________________________________________________________________________

    def get_data_loader(
        self,
        batch_size=32,
        shuffle=True,
        imbalance_data=False,
        return_imbalance_weight=False,
    ):
        if imbalance_data:
            labels = self.metadata_df[self.label_col].values
            label_counts = pd.Series(labels).value_counts()
            class_weights = 1.0 / label_counts
            sample_weights = [class_weights[label] for label in labels]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            data_loader = DataLoader(self, batch_size=batch_size, sampler=sampler)
            if return_imbalance_weight:
                return data_loader, class_weights
            else:
                return data_loader
        else:
            data_loader = DataLoader(self, batch_size=batch_size, shuffle=shuffle)
        return data_loader

    # __________________________________________________________________________

    def calculate_mean_std(self, channel_dim=211, verbose=False):
        mean_sum = torch.zeros(channel_dim)
        squared_sum = torch.zeros(channel_dim)
        total_pixels, count = 0, 0
        for images, _ in self:
            if len(images.shape) == 2:
                count += 1
                total_pixels += images.size(0)
                mean_sum += images.sum(dim=0)
                squared_sum += (images**2).sum(dim=0)
            else:
                raise Exception(f"Data shape not supported: {images.shape}")
        if verbose:
            print("Total images: ", count)
        mean = mean_sum / total_pixels
        std = torch.sqrt(squared_sum / total_pixels - mean**2)
        return mean, std
