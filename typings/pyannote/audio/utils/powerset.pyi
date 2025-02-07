from functools import cached_property as cached_property

import torch
import torch.nn as nn
from _typeshed import Incomplete

class Powerset(nn.Module):
    num_classes: Incomplete
    max_set_size: Incomplete
    def __init__(self, num_classes: int, max_set_size: int) -> None: ...
    @cached_property
    def num_powerset_classes(self) -> int: ...
    def build_mapping(self) -> torch.Tensor: ...
    def build_cardinality(self) -> torch.Tensor: ...
    def to_multilabel(self, powerset: torch.Tensor) -> torch.Tensor: ...
    def forward(self, powerset: torch.Tensor) -> torch.Tensor: ...
    def to_powerset(self, multilabel: torch.Tensor) -> torch.Tensor: ...
