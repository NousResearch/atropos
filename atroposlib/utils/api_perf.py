"""
API performance tracker for trainer-inference communication optimization.

Provides lightweight latency and throughput monitoring for the scored_data
API round-trip, enabling bottleneck identification in the trainer-inference
communication pipeline.

Features:
- Rolling window latency tracking (configurable window size)
- Throughput computation (items/sec, requests/sec)
- Percentile latency statistics (p50, p95, p99)
- Compression ratio tracking
- WandB-compatible metrics output

Usage:
    tracker = APIPerformanceTracker(window_size=100)

    # Around API call
    with tracker.track_request(n_items=group_size, payload_bytes=len(data)):
        await send_scored_data(...)

    # Log to wandb
    wandb.log(tracker.metrics_dict())
"""

import logging
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RequestRecord:
    """Record of a single API request."""

    latency_ms: float
    n_items: int
    payload_bytes: int
    compressed_bytes: int
    timestamp: float
    success: bool = True


class APIPerformanceTracker:
    """
    Lightweight performance tracker for trainer-inference API communication.

    Maintains a rolling window of request records for computing latency
    and throughput statistics without unbounded memory growth.

    Args:
        window_size: Number of recent requests to keep for stats. Default: 200.
        slow_request_threshold_ms: Latency above this triggers a warning. Default: 5000.
    """

    def __init__(
        self,
        window_size: int = 200,
        slow_request_threshold_ms: float = 5000.0,
    ):
        self.window_size = max(1, window_size)
        self.slow_request_threshold_ms = slow_request_threshold_ms
        self._records: deque = deque(maxlen=self.window_size)
        self._total_requests: int = 0
        self._total_items: int = 0
        self._total_bytes_sent: int = 0
        self._total_compressed_bytes: int = 0
        self._failed_requests: int = 0

    @contextmanager
    def track_request(
        self,
        n_items: int = 1,
        payload_bytes: int = 0,
        compressed_bytes: int = 0,
    ):
        """
        Context manager to track a single API request.

        Args:
            n_items: Number of items (completions) in this request.
            payload_bytes: Size of the uncompressed payload.
            compressed_bytes: Size of the compressed payload (0 if no compression).

        Yields:
            None. Timing is handled automatically.
        """
        start = time.monotonic()
        success = True
        try:
            yield
        except Exception:
            success = False
            self._failed_requests += 1
            raise
        finally:
            elapsed_ms = (time.monotonic() - start) * 1000.0

            record = RequestRecord(
                latency_ms=elapsed_ms,
                n_items=n_items,
                payload_bytes=payload_bytes,
                compressed_bytes=compressed_bytes if compressed_bytes > 0 else payload_bytes,
                timestamp=time.time(),
                success=success,
            )
            self._records.append(record)
            self._total_requests += 1
            self._total_items += n_items
            self._total_bytes_sent += payload_bytes
            self._total_compressed_bytes += (
                compressed_bytes if compressed_bytes > 0 else payload_bytes
            )

            if elapsed_ms > self.slow_request_threshold_ms:
                logger.warning(
                    "Slow API request: %.1fms (threshold: %.1fms, items: %d)",
                    elapsed_ms,
                    self.slow_request_threshold_ms,
                    n_items,
                )

    def record_request(
        self,
        latency_ms: float,
        n_items: int = 1,
        payload_bytes: int = 0,
        compressed_bytes: int = 0,
        success: bool = True,
    ):
        """
        Manually record a request (for cases where context manager isn't suitable).

        Args:
            latency_ms: Request latency in milliseconds.
            n_items: Number of items in the request.
            payload_bytes: Uncompressed payload size.
            compressed_bytes: Compressed payload size.
            success: Whether the request succeeded.
        """
        record = RequestRecord(
            latency_ms=latency_ms,
            n_items=n_items,
            payload_bytes=payload_bytes,
            compressed_bytes=compressed_bytes if compressed_bytes > 0 else payload_bytes,
            timestamp=time.time(),
            success=success,
        )
        self._records.append(record)
        self._total_requests += 1
        self._total_items += n_items
        self._total_bytes_sent += payload_bytes
        self._total_compressed_bytes += (
            compressed_bytes if compressed_bytes > 0 else payload_bytes
        )
        if not success:
            self._failed_requests += 1

        if latency_ms > self.slow_request_threshold_ms:
            logger.warning(
                "Slow API request: %.1fms (threshold: %.1fms, items: %d)",
                latency_ms,
                self.slow_request_threshold_ms,
                n_items,
            )

    @property
    def n_records(self) -> int:
        """Number of records in the rolling window."""
        return len(self._records)

    def latency_stats(self) -> Dict[str, float]:
        """
        Compute latency statistics from the rolling window.

        Returns:
            Dictionary with p50, p95, p99, mean, min, max latencies in ms.
        """
        if not self._records:
            return {
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "mean_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
            }

        latencies = np.array([r.latency_ms for r in self._records])
        return {
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "mean_ms": float(np.mean(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
        }

    def throughput_stats(self) -> Dict[str, float]:
        """
        Compute throughput statistics from the rolling window.

        Returns:
            Dictionary with items/sec and requests/sec over the window.
        """
        if len(self._records) < 2:
            return {
                "items_per_sec": 0.0,
                "requests_per_sec": 0.0,
            }

        records = list(self._records)
        time_span = records[-1].timestamp - records[0].timestamp
        if time_span <= 0:
            return {
                "items_per_sec": 0.0,
                "requests_per_sec": 0.0,
            }

        total_items = sum(r.n_items for r in records)
        return {
            "items_per_sec": total_items / time_span,
            "requests_per_sec": len(records) / time_span,
        }

    def compression_stats(self) -> Dict[str, float]:
        """
        Compute compression statistics from the rolling window.

        Returns:
            Dictionary with mean compression ratio and total bytes sent.
        """
        if not self._records:
            return {
                "mean_compression_ratio": 1.0,
                "mean_payload_bytes": 0.0,
            }

        ratios = []
        payloads = []
        for r in self._records:
            if r.payload_bytes > 0:
                ratios.append(r.compressed_bytes / r.payload_bytes)
            payloads.append(float(r.payload_bytes))

        return {
            "mean_compression_ratio": float(np.mean(ratios)) if ratios else 1.0,
            "mean_payload_bytes": float(np.mean(payloads)),
        }

    def metrics_dict(self) -> Dict[str, float]:
        """
        Return all performance metrics for WandB logging.

        Returns:
            Dictionary with keys prefixed by 'api_perf/' for clean namespacing.
        """
        metrics = {}

        latency = self.latency_stats()
        for key, val in latency.items():
            metrics[f"api_perf/latency_{key}"] = val

        throughput = self.throughput_stats()
        for key, val in throughput.items():
            metrics[f"api_perf/{key}"] = val

        compression = self.compression_stats()
        for key, val in compression.items():
            metrics[f"api_perf/{key}"] = val

        metrics["api_perf/total_requests"] = float(self._total_requests)
        metrics["api_perf/total_items"] = float(self._total_items)
        metrics["api_perf/failed_requests"] = float(self._failed_requests)
        metrics["api_perf/error_rate"] = (
            self._failed_requests / max(1, self._total_requests)
        )

        return metrics

    def reset(self):
        """Clear all records and counters."""
        self._records.clear()
        self._total_requests = 0
        self._total_items = 0
        self._total_bytes_sent = 0
        self._total_compressed_bytes = 0
        self._failed_requests = 0
