from torch.optim import Optimizer as Optimizer
from typing import Optional, Text

def ReduceLROnPlateau(optimizer: Optimizer, monitor: Optional[Text] = ..., direction: Optional[Text] = ..., min_lr: float = ..., max_lr: float = ..., factor: float = ..., patience: int = ..., **kwargs): ...
