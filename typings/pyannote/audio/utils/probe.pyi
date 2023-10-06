import torch.nn as nn
from typing import Callable, Dict, Text

def probe(trunk: nn.Module, branches: Dict[Text, Text]) -> Callable: ...
