from typing import Optional

from omegaconf import DictConfig as DictConfig
from pyannote.database import ProtocolFile as ProtocolFile

def evaluate(cfg: DictConfig) -> Optional[float]: ...
