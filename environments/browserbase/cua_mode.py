from __future__ import annotations


class CUAModeAdapter:
    def ensure_supported(self) -> None:
        raise NotImplementedError(
            "Browserbase CUA mode is not implemented in Atropos yet. "
            "The runtime surface is reserved, but screenshot-bearing multimodal "
            "token accounting needs to land before CUA rollouts are enabled."
        )
