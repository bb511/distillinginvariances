# Logger that prefixes every message with the rank of the process that emitted it.

import logging
from collections.abc import Mapping

from lightning_utilities.core.rank_zero import rank_prefixed_message, rank_zero_only


class RankedLogger(logging.LoggerAdapter):
    """A logger for multi-process runs, prefixing each message with the rank."""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        """Log on every process, or on rank 0 alone when *rank_zero_only* is set."""
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def _emits(self, current_rank: int, rank: int | None) -> bool:
        """Whether this process logs the message: rank 0 alone, or the rank *rank*
        names."""
        if self.rank_zero_only:
            return current_rank == 0
        return rank is None or current_rank == rank

    def log(
        self, level: int, msg: str, *args, rank: int | None = None, **kwargs
    ) -> None:
        """Log *msg* at *level*, prefixed with the rank of the process it is logged
        from.

        :param rank: log on that process alone; ``None`` logs on every process, subject
            to ``rank_zero_only``. Keyword-only, so ``log.info("%s", value)`` formats as
            logging does.
        :raises RuntimeError: before Lightning has set ``rank_zero_only.rank``, without
            which the rank of this process is unknown.
        """
        if not self.isEnabledFor(level):
            return
        msg, kwargs = self.process(msg, kwargs)
        current_rank = getattr(rank_zero_only, "rank", None)
        if current_rank is None:
            raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
        if self._emits(current_rank, rank):
            self.logger.log(
                level, rank_prefixed_message(msg, current_rank), *args, **kwargs
            )
