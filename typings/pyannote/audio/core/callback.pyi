from typing import Mapping, Union

from _typeshed import Incomplete
from pyannote.audio import Model as Model
from pytorch_lightning import Callback
from pytorch_lightning import Trainer as Trainer

class GraduallyUnfreeze(Callback):
    epochs_per_stage: Incomplete
    schedule: Incomplete
    def __init__(
        self,
        schedule: Union[Mapping[str, int], list[Union[list[str], str]]] = ...,
        epochs_per_stage: int = ...,
    ) -> None: ...
    def on_fit_start(self, trainer: Trainer, model: Model): ...
    def on_train_epoch_start(self, trainer: Trainer, model: Model): ...
