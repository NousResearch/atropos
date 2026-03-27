"""
Tests for API performance tracking utilities.

Tests cover:
- Request tracking via context manager
- Latency percentile computation (p50, p95, p99)
- Throughput calculation (items/sec, requests/sec)
- Compression ratio tracking
- Rolling window behavior
- Multi-threaded/Parallel safety (simulated via multiple records)
- WandB metrics dictionary formatting
"""

import time
import unittest
from unittest.mock import patch
import numpy as np
from atroposlib.utils.api_perf import APIPerformanceTracker

class TestAPIPerformanceTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = APIPerformanceTracker(window_size=10)

    def test_track_request_context_manager(self):
        with patch('time.monotonic', side_effect=[0, 0.05]): # 50ms latency
            with self.tracker.track_request(n_items=2, payload_bytes=1000):
                pass
        
        stats = self.tracker.latency_stats()
        self.assertEqual(self.tracker.n_records, 1)
        self.assertAlmostEqual(stats['mean_ms'], 50.0)

    def test_latency_percentiles(self):
        # Record 10 requests with specific latencies: 10, 20, ..., 100ms
        for i in range(1, 11):
            self.tracker.record_request(latency_ms=float(i * 10), n_items=1)
        
        stats = self.tracker.latency_stats()
        # p50 of [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] is 55.0
        self.assertAlmostEqual(stats['p50_ms'], 55.0)
        # p95 is between 90 and 100
        self.assertGreater(stats['p95_ms'], 90.0)
        self.assertLessEqual(stats['p95_ms'], 100.0)

    def test_rolling_window(self):
        tracker = APIPerformanceTracker(window_size=3)
        for i in range(5):
            tracker.record_request(latency_ms=float(i), n_items=1)
        
        self.assertEqual(tracker.n_records, 3)
        stats = tracker.latency_stats()
        # Should only have records for 2, 3, 4
        self.assertAlmostEqual(stats['min_ms'], 2.0)
        self.assertAlmostEqual(stats['max_ms'], 4.0)

    def test_throughput_calculation(self):
        # Record 2 requests at t=0 and t=1s
        with patch('time.time', side_effect=[100.0, 101.0]):
            self.tracker.record_request(latency_ms=10, n_items=10) # t=100
            self.tracker.record_request(latency_ms=10, n_items=10) # t=101
        
        stats = self.tracker.throughput_stats()
        # 20 items over 1 second = 20 items/sec
        self.assertAlmostEqual(stats['items_per_sec'], 20.0)
        # 2 requests over 1 second = 2 req/sec
        self.assertAlmostEqual(stats['requests_per_sec'], 2.0)

    def test_compression_stats(self):
        # 50% compression
        self.tracker.record_request(latency_ms=10, payload_bytes=1000, compressed_bytes=500)
        # 25% compression
        self.tracker.record_request(latency_ms=10, payload_bytes=1000, compressed_bytes=250)
        
        stats = self.tracker.compression_stats()
        # mean of 0.5 and 0.25 is 0.375
        self.assertAlmostEqual(stats['mean_compression_ratio'], 0.375)

    def test_metrics_dict_formatting(self):
        self.tracker.record_request(latency_ms=50, n_items=5, success=True)
        self.tracker.record_request(latency_ms=100, n_items=5, success=False)
        
        metrics = self.tracker.metrics_dict()
        self.assertIn('api_perf/latency_p50_ms', metrics)
        self.assertIn('api_perf/items_per_sec', metrics)
        self.assertEqual(metrics['api_perf/failed_requests'], 1)
        self.assertEqual(metrics['api_perf/error_rate'], 0.5)

    def test_slow_request_warning(self):
        with self.assertLogs('atroposlib.utils.api_perf', level='WARNING') as cm:
            self.tracker.slow_request_threshold_ms = 100
            self.tracker.record_request(latency_ms=150)
        self.assertTrue(any("Slow API request" in msg for msg in cm.output))

if __name__ == '__main__':
    unittest.main()
