from typing import Callable

import torch.nn as nn

def probe(trunk: nn.Module, branches: dict[str, str]) -> Callable: ...
