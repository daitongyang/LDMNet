# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
import mmengine.fileio as fileio

@DATASETS.register_module()
class LaRSDataset(BaseSegDataset):

    METAINFO = dict(
        classes=(
           'Static Obstacle','Water','Sky'),
        palette=[[247, 195, 37], [41, 167, 224], [90, 75, 164]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, reduce_zero_label=reduce_zero_label,**kwargs)
        assert fileio.exists(
                    self.data_prefix['img_path'], backend_args=self.backend_args)