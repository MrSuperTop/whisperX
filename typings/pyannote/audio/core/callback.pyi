from _typeshed import Incomplete
from pyannote.audio import Model as Model
from pytorch_lightning import Callback, Trainer as Trainer
from typing import List, Mapping, Text, Union

class GraduallyUnfreeze(Callback):
    epochs_per_stage: Incomplete
    schedule: Incomplete
    def __init__(self, schedule: Union[Mapping[Text, int], List[Union[List[Text], Text]]] = ..., epochs_per_stage: int = ...) -> None: ...
    def on_fit_start(self, trainer: Trainer, model: Model): ...
    def on_train_epoch_start(self, trainer: Trainer, model: Model): ...
