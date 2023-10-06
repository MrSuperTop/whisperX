from _typeshed import Incomplete
from enum import Enum
from pyannote.database import Database as Database
from typing import Text

app: Incomplete

class Task(str, Enum):
    Any: str
    Protocol: str
    Collection: str
    SpeakerDiarization: str
    SpeakerVerification: str

def database() -> None: ...
def task(database: str = ...): ...
def protocol(database: str = ..., task: Task = ...): ...
def duration_to_str(seconds: float) -> Text: ...
def info(protocol: str): ...
def main() -> None: ...
