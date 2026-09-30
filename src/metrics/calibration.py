"""Expected calibration error, accumulated over the batches of one split."""

import torch


def class_indices(targets: torch.Tensor) -> torch.Tensor:
    """The class index of each row; the datamodules serve one-hot targets."""
    return targets if targets.ndim == 1 else targets.argmax(-1)


class ECEAccumulator:
    """Equal-width-binned ECE (Guo et al.

    2017, arXiv:1706.04599, eq. 3).
        The bin sums are float64: a 260k-row test split accumulated in float32 loses the last
        digits of the counts, and the study quotes the third.
    """

    def __init__(self, n_bins: int = 15) -> None:
        self.n_bins = n_bins
        self._count = torch.zeros(n_bins, dtype=torch.float64)
        self._confidence = torch.zeros(n_bins, dtype=torch.float64)
        self._correct = torch.zeros(n_bins, dtype=torch.float64)

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        confidence, prediction = logits.detach().softmax(-1).max(-1)
        confidence = confidence.double().cpu()
        # Bin b covers [b/n, (b+1)/n); a confidence of exactly 1 belongs to the last one.
        index = (confidence * self.n_bins).long().clamp(max=self.n_bins - 1)
        hit = prediction.cpu().eq(class_indices(targets).cpu()).double()
        self._count.scatter_add_(0, index, torch.ones_like(confidence))
        self._confidence.scatter_add_(0, index, confidence)
        self._correct.scatter_add_(0, index, hit)

    def compute(self) -> float:
        """Sum over the filled bins of ``n_b / n |acc_b - conf_b|``; NaN before any update."""
        total = float(self._count.sum())
        if total == 0.0:
            return float("nan")
        filled = self._count > 0.0
        # n_b |acc_b - conf_b| is the gap between the two bin sums, so neither is divided.
        gap = (self._correct[filled] - self._confidence[filled]).abs()
        return float((gap / total).sum())
