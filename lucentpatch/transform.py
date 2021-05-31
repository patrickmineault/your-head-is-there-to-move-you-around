# Copyright 2020 The Lucent Authors. All Rights Reserved.
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
# ==============================================================================

import torch
from torchvision.transforms import Normalize
import numpy as np


def normalize():
    # Kinetics-400 normalization for torchvision models
    # see https://pytorch.org/docs/stable/torchvision/models.html
    normal = Normalize(
        mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]
    )

    def inner(image_t):
        output = torch.stack([normal(t) for t in image_t])
        return output

    return inner


def loop(nt):
    dts = np.arange(nt) - (nt - 1) // 2

    def inner(image_t):
        dt = np.random.choice(dts)
        if dt == 0:
            return image_t
        elif dt > 0:
            return torch.cat((image_t[dt:, ...], 0 * image_t[:dt, ...]), dim=0)
        else:
            dt = abs(dt)
            return torch.cat((0 * image_t[-dt:, ...], image_t[:-dt, ...]), dim=0)

    return inner
