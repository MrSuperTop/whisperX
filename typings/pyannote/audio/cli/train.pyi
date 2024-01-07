from typing import Optional

from omegaconf import DictConfig as DictConfig

def train(cfg: DictConfig) -> Optional[float]: ...
