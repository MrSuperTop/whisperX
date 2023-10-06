from omegaconf import DictConfig as DictConfig
from pyannote.database import ProtocolFile as ProtocolFile
from typing import Optional

def evaluate(cfg: DictConfig) -> Optional[float]: ...
