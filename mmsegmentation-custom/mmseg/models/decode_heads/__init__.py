# Copyright (c) OpenMMLab. All rights reserved.
from .vw_head import VWHead, VWCityHead
from .fcn_head import FCNHead
from .nl_head import NLHead

__all__ = ['VWHead', 'VWCityHead', 'FCNHead', 'NLHead']
