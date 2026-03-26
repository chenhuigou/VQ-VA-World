# Copyright 2024-2025 The Alibaba Wan Team Authors.
# Copyright 2025 Zeyu Wang & Zilong Chen.
# SPDX-License-Identifier: Apache-2.0

import copy
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from .wan_ti2v_5B import ti2v_5B

WAN_CONFIGS = {
    'ti2v-5B': ti2v_5B,
}

SIZE_CONFIGS = {
    '480*832': (480, 832),
    '832*480': (832, 480),
}

MAX_AREA_CONFIGS = {
    '480*832': 480 * 832,
    '832*480': 832 * 480,
}

SUPPORTED_SIZES = {
    'ti2v-5B': ('704*1280', '1280*704'),
}
