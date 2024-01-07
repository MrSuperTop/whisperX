from typing import Optional

from numpy.typing import ArrayLike as ArrayLike

def plot_distributions(
    y_true: ArrayLike,
    scores: ArrayLike,
    save_to: str,
    xlim: Optional[float, float] = ...,
    nbins: int = ...,
    ymax: float = ...,
    dpi: int = ...,
) -> bool: ...
def plot_det_curve(
    y_true: ArrayLike,
    scores: ArrayLike,
    save_to: str,
    distances: bool = ...,
    dpi: int = ...,
) -> float: ...
def plot_precision_recall_curve(
    y_true: ArrayLike,
    scores: ArrayLike,
    save_to: str,
    distances: bool = ...,
    dpi: int = ...,
) -> float: ...
