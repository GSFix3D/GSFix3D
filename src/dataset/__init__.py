# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# More information about Marigold:
#   https://marigoldmonodepth.github.io
#   https://marigoldcomputervision.github.io
# Efficient inference pipelines are now part of diffusers:
#   https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage
#   https://huggingface.co/docs/diffusers/api/pipelines/marigold
# Examples of trained models and live demos:
#   https://huggingface.co/prs-eth
# Related projects:
#   https://rollingdepth.github.io/
#   https://marigolddepthcompletion.github.io/
# Citation (BibTeX):
#   https://github.com/prs-eth/Marigold#-citation
# If you find Marigold useful, we kindly ask you to cite our papers.
# --------------------------------------------------------------------------
# This file has been modified by Mobile Robotics Lab, TU Munich (2025).
# SPDX-FileCopyrightText: 2025 Jiaxin Wei
# SPDX-License-Identifier: Apache-2.0


import os
from typing import Union, List

from .base_dataset import BaseTrainDataset, BaseFinetuneDataset, DatasetMode
from .vkitti_dataset import VirtualKITTIDataset


dataset_name_class_dict = {
    "hypersim_gsfixer": BaseTrainDataset,
    "vkitti_gsfixer": VirtualKITTIDataset,
    "replica": BaseFinetuneDataset,
    "scannetpp": BaseFinetuneDataset,
}


def get_dataset(
    cfg_data_split, base_data_dir: str, mode: DatasetMode, **kwargs
) -> Union[
    BaseTrainDataset, BaseFinetuneDataset,
    List[BaseTrainDataset], List[BaseFinetuneDataset],
]:
    if "mixed" == cfg_data_split.name:
        assert DatasetMode.TRAIN == mode, "Only training mode supports mixed datasets."
        dataset_ls = [
            get_dataset(_cfg, base_data_dir, mode, **kwargs)
            for _cfg in cfg_data_split.dataset_list
        ]
        return dataset_ls
    elif cfg_data_split.name in dataset_name_class_dict.keys():
        dataset_class = dataset_name_class_dict[cfg_data_split.name]
        dataset = dataset_class(
            mode=mode,
            dataset_dir=os.path.join(base_data_dir, cfg_data_split.dir),
            **cfg_data_split,
            **kwargs,
        )
    else:
        raise NotImplementedError

    return dataset
