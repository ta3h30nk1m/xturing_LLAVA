from dataclasses import dataclass
from pathlib import Path
from typing import Union

from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk

import glob, random, jsonlines, os

from xturing.datasets.base import BaseDataset

class ExampleDataset(BaseDataset):
    config_name: str = "example_dataset"

    def __init__(self, path: Union[str, Path, HFDataset, dict]):
        if isinstance(path, HFDataset) or isinstance(path, DatasetDict):
            self.data = path
        elif isinstance(path, dict):
            self.data = {"train": HFDataset.from_dict(path)}
        else:
            assert Path(path).exists(), "path does not exist"
            self.data = []
            laion_files = glob.glob(os.path.join(path,'*.jsonl'))
            for file in laion_files:
                with jsonlines.open(file) as f:
                    for line in f.iter():
                        self.data.append({'img': random.choice(line['image_info'])['raw_url'], 'caption': random.choice(line['text_list'])})
        self._validate()
        self._meta = TextDatasetMeta()
        self._template = None

    def _validate(self):
        # check is hf dataset has train split and if it has column text, and if there are any other - it should be target
        assert "train" in self.data, "The dataset should have a train split"
        assert (
            "text" in self.data["train"].column_names
        ), "The dataset should have a column named text"

        if len(self.data["train"].column_names) > 1:
            assert (
                "target" in self.data["train"].column_names
            ), "The dataset should have a column named target if there is more than one column"
            assert (
                len(self.data["train"].column_names) == 2
            ), "The dataset should have only two columns, text and target"

    def __len__(self):
        return len(self.data["train"])

    def __iter__(self):
        return iter(self.data["train"])

    def __getitem__(self, idx):
        return self.data["train"][idx]

    def save(self, path):
        return self.data.save_to_disk(path)
