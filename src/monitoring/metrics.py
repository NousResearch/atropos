"""
Metrics collection and monitoring for LLM Society Simulation
"""

import asyncio
import json
import logging
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MetricEntry:
    """Single metric entry"""

    timestamp: float
    step: int
    metric_name: str
    value: float
    tags: Dict[str, str]


class MetricsCollector:
    """
    Collects and stores simulation metrics
    """

    def __init__(self, config):
        self.config = config

        # Storage
        self.metrics: List[MetricEntry] = []
        self.db_path = Path(config.output.directory) / "metrics.db"
        self.db_lock = threading.Lock()

        # Running state
        self._running = False
        self._storage_task: Optional[asyncio.Task] = None

        # Ensure output directory exists
        Path(config.output.directory).mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        logger.info(f"Metrics collector initialized, storing to: {self.db_path}")

    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        step INTEGER NOT NULL,
                        metric_name TEXT NOT NULL,
                        value REAL NOT NULL,
                        tags TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                # Create indexes
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_timestamp ON metrics(timestamp)"
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_step ON metrics(step)")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_metric_name ON metrics(metric_name)"
                )

                conn.commit()

        except Exception as e:
            logger.error(f"Failed to initialize metrics database: {e}")

    async def start(self):
        """Start metrics collection"""
        if self._running:
            return

        self._running = True
        self._storage_task = asyncio.create_task(self._storage_worker())
        logger.info("Metrics collection started")

    async def stop(self):
        """Stop metrics collection and flush remaining data"""
        self._running = False

        if self._storage_task:
            self._storage_task.cancel()
            try:
                await self._storage_task
            except asyncio.CancelledError:
                pass

        # Flush any remaining metrics
        await self._flush_metrics()
        logger.info("Metrics collection stopped")

    async def record_metrics(self, data: Dict[str, Any]):
        """Record simulation metrics"""
        try:
            timestamp = data.get("timestamp", time.time())
            step = data.get("current_step", 0)

            # Extract basic simulation metrics
            basic_metrics = [
                ("steps_per_second", data.get("steps_per_second", 0.0)),
                ("agent_count", data.get("agent_count", 0)),
                ("total_social_interactions", data.get("total_social_interactions", 0)),
                ("total_objects_created", data.get("total_objects_created", 0)),
                ("avg_energy", data.get("avg_energy", 0.0)),
                ("avg_happiness", data.get("avg_happiness", 0.0)),
                ("runtime", data.get("runtime", 0.0)),
            ]

            # Record basic metrics
            for metric_name, value in basic_metrics:
                if value is not None:
                    entry = MetricEntry(
                        timestamp=timestamp,
                        step=step,
                        metric_name=metric_name,
                        value=float(value),
                        tags={"category": "simulation"},
                    )
                    self.metrics.append(entry)

            # Record LLM metrics if available
            llm_stats = data.get("llm_stats", {})
            if llm_stats:
                llm_metrics = [
                    ("llm_total_requests", llm_stats.get("total_requests", 0)),
                    ("llm_cached_responses", llm_stats.get("cached_responses", 0)),
                    ("llm_failed_requests", llm_stats.get("failed_requests", 0)),
                    ("llm_cache_hit_rate", llm_stats.get("cache_hit_rate", 0.0)),
                    ("llm_failure_rate", llm_stats.get("failure_rate", 0.0)),
                    ("llm_cache_size", llm_stats.get("cache_size", 0)),
                ]

                for metric_name, value in llm_metrics:
                    if value is not None:
                        entry = MetricEntry(
                            timestamp=timestamp,
                            step=step,
                            metric_name=metric_name,
                            value=float(value),
                            tags={"category": "llm"},
                        )
                        self.metrics.append(entry)

        except Exception as e:
            logger.error(f"Error recording metrics: {e}")

    async def _storage_worker(self):
        """Background worker to periodically store metrics"""
        while self._running:
            try:
                await asyncio.sleep(5.0)  # Store every 5 seconds
                await self._flush_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics storage worker: {e}")
                await asyncio.sleep(1.0)

    async def _flush_metrics(self):
        """Flush accumulated metrics to storage"""
        if not self.metrics:
            return

        try:
            # Copy and clear metrics list
            metrics_to_store = self.metrics.copy()
            self.metrics.clear()

            # Store to database in a thread to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                None, self._store_metrics_to_db, metrics_to_store
            )

            logger.debug(f"Stored {len(metrics_to_store)} metrics to database")

        except Exception as e:
            logger.error(f"Error flushing metrics: {e}")

    def _store_metrics_to_db(self, metrics: List[MetricEntry]):
        """Store metrics to SQLite database (runs in thread)"""
        try:
            with self.db_lock:
                with sqlite3.connect(self.db_path) as conn:
                    for metric in metrics:
                        conn.execute(
                            """
                            INSERT INTO metrics (timestamp, step, metric_name, value, tags)
                            VALUES (?, ?, ?, ?, ?)
                        """,
                            (
                                metric.timestamp,
                                metric.step,
                                metric.metric_name,
                                metric.value,
                                json.dumps(metric.tags),
                            ),
                        )
                    conn.commit()

        except Exception as e:
            logger.error(f"Error storing metrics to database: {e}")

    def export_metrics(
        self, format: str = "json", output_path: Optional[str] = None
    ) -> str:
        """Export collected metrics to file"""
        if not output_path:
            timestamp = int(time.time())
            output_path = (
                Path(self.config.output.directory) / f"metrics_{timestamp}.{format}"
            )
        else:
            output_path = Path(output_path)

        try:
            # Query all metrics from database
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT timestamp, step, metric_name, value, tags
                    FROM metrics
                    ORDER BY timestamp, step
                """
                )

                metrics_data = []
                for row in cursor:
                    metrics_data.append(
                        {
                            "timestamp": row["timestamp"],
                            "step": row["step"],
                            "metric_name": row["metric_name"],
                            "value": row["value"],
                            "tags": json.loads(row["tags"]) if row["tags"] else {},
                        }
                    )

            # Export based on format
            if format.lower() == "json":
                with open(output_path, "w") as f:
                    json.dump(metrics_data, f, indent=2)

            elif format.lower() == "csv":
                import pandas as pd

                df = pd.DataFrame(metrics_data)
                df.to_csv(output_path, index=False)

            else:
                raise ValueError(f"Unsupported export format: {format}")

            logger.info(f"Exported {len(metrics_data)} metrics to {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            raise

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of collected metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT
                        COUNT(*) as total_entries,
                        MIN(timestamp) as start_time,
                        MAX(timestamp) as end_time,
                        COUNT(DISTINCT step) as unique_steps,
                        COUNT(DISTINCT metric_name) as unique_metrics
                    FROM metrics
                """
                )

                row = cursor.fetchone()

                return {
                    "total_entries": row[0],
                    "start_time": row[1],
                    "end_time": row[2],
                    "duration": row[2] - row[1] if row[1] and row[2] else 0,
                    "unique_steps": row[3],
                    "unique_metrics": row[4],
                }

        except Exception as e:
            logger.error(f"Error getting summary stats: {e}")
            return {}

    def get_metric_history(
        self, metric_name: str, limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get history for a specific metric"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT timestamp, step, value, tags
                    FROM metrics
                    WHERE metric_name = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (metric_name, limit),
                )

                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Error getting metric history: {e}")
            return []
