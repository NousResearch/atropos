"""
Database Manager for CloudVR-PerfGuard
Handles storage and retrieval of test jobs, performance data, and regression analysis data
"""

import asyncio
import json
import sqlite3
import aiosqlite
from typing import Dict, List, Any, Optional
from datetime import datetime
import os

class DatabaseManager:
    """
    Manages database operations for CloudVR-PerfGuard
    Uses SQLite for MVP, can be upgraded to PostgreSQL/Cloud SQL later
    """
    
    def __init__(self, db_path: str = "cloudvr_perfguard.db"):
        self.db_path = db_path
        self.connection = None
        
    async def initialize(self):
        """Initialize database and create tables"""
        
        try:
            # Create database directory if it doesn't exist
            os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
            
            # Connect to database
            self.connection = await aiosqlite.connect(self.db_path)
            
            # Create tables
            await self._create_tables()
            
            print(f"INFO: Database initialized at {self.db_path}")
            
        except Exception as e:
            print(f"ERROR: Failed to initialize database: {e}")
            raise
    
    async def _create_tables(self):
        """Create database tables"""
        
        # Test jobs table
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS test_jobs (
                job_id TEXT PRIMARY KEY,
                app_name TEXT NOT NULL,
                build_version TEXT NOT NULL,
                platform TEXT NOT NULL,
                submission_type TEXT NOT NULL,
                baseline_version TEXT,
                build_path TEXT NOT NULL,
                test_config TEXT,
                status TEXT DEFAULT 'queued',
                progress INTEGER DEFAULT 0,
                estimated_completion TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # Performance results table
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS performance_results (
                job_id TEXT PRIMARY KEY,
                test_id TEXT NOT NULL,
                build_path TEXT NOT NULL,
                config TEXT NOT NULL,
                total_duration REAL NOT NULL,
                individual_results TEXT NOT NULL,
                aggregated_metrics TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES test_jobs (job_id)
            )
        """)
        
        # Regression analysis table
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS regression_analysis (
                job_id TEXT PRIMARY KEY,
                regressions TEXT NOT NULL,
                statistical_analysis TEXT,
                comparison TEXT NOT NULL,
                regression_score TEXT NOT NULL,
                overall_status TEXT NOT NULL,
                analysis_timestamp TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES test_jobs (job_id)
            )
        """)
        
        # Create indexes for better performance
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_test_jobs_app_version 
            ON test_jobs (app_name, build_version)
        """)
        
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_test_jobs_status 
            ON test_jobs (status)
        """)
        
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_test_jobs_submission_type 
            ON test_jobs (submission_type)
        """)
        
        await self.connection.commit()
    
    async def create_test_job(
        self,
        job_id: str,
        app_name: str,
        build_version: str,
        platform: str,
        submission_type: str,
        build_path: str,
        test_config: Dict[str, Any],
        baseline_version: Optional[str] = None
    ) -> bool:
        """Create a new test job record"""
        
        try:
            now = datetime.utcnow().isoformat()
            
            await self.connection.execute("""
                INSERT INTO test_jobs (
                    job_id, app_name, build_version, platform, submission_type,
                    baseline_version, build_path, test_config, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)
            """, (
                job_id, app_name, build_version, platform, submission_type,
                baseline_version, build_path, json.dumps(test_config), now, now
            ))
            
            await self.connection.commit()
            return True
            
        except Exception as e:
            print(f"ERROR creating test job {job_id}: {e}")
            return False
    
    async def get_test_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get test job by ID"""
        
        try:
            cursor = await self.connection.execute("""
                SELECT * FROM test_jobs WHERE job_id = ?
            """, (job_id,))
            
            row = await cursor.fetchone()
            if not row:
                return None
            
            # Convert row to dictionary
            columns = [description[0] for description in cursor.description]
            job_data = dict(zip(columns, row))
            
            # Parse JSON fields
            if job_data.get("test_config"):
                job_data["test_config"] = json.loads(job_data["test_config"])
            
            return job_data
            
        except Exception as e:
            print(f"ERROR getting test job {job_id}: {e}")
            return None
    
    async def update_job_status(
        self, 
        job_id: str, 
        status: str, 
        progress: Optional[int] = None,
        error: Optional[str] = None
    ) -> bool:
        """Update job status and progress"""
        
        try:
            now = datetime.utcnow().isoformat()
            
            if progress is not None:
                await self.connection.execute("""
                    UPDATE test_jobs 
                    SET status = ?, progress = ?, error_message = ?, updated_at = ?
                    WHERE job_id = ?
                """, (status, progress, error, now, job_id))
            else:
                await self.connection.execute("""
                    UPDATE test_jobs 
                    SET status = ?, error_message = ?, updated_at = ?
                    WHERE job_id = ?
                """, (status, error, now, job_id))
            
            await self.connection.commit()
            return True
            
        except Exception as e:
            print(f"ERROR updating job status for {job_id}: {e}")
            return False
    
    async def store_performance_results(
        self, 
        job_id: str, 
        results: Dict[str, Any]
    ) -> bool:
        """Store performance test results"""
        
        try:
            await self.connection.execute("""
                INSERT OR REPLACE INTO performance_results (
                    job_id, test_id, build_path, config, total_duration,
                    individual_results, aggregated_metrics, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job_id,
                results.get("test_id", ""),
                results.get("build_path", ""),
                json.dumps(results.get("config", {})),
                results.get("total_duration", 0),
                json.dumps(results.get("individual_results", [])),
                json.dumps(results.get("aggregated_metrics", {})),
                results.get("timestamp", datetime.utcnow().isoformat())
            ))
            
            await self.connection.commit()
            return True
            
        except Exception as e:
            print(f"ERROR storing performance results for {job_id}: {e}")
            return False
    
    async def get_performance_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get performance results by job ID"""
        
        try:
            cursor = await self.connection.execute("""
                SELECT * FROM performance_results WHERE job_id = ?
            """, (job_id,))
            
            row = await cursor.fetchone()
            if not row:
                return None
            
            # Convert row to dictionary
            columns = [description[0] for description in cursor.description]
            results = dict(zip(columns, row))
            
            # Parse JSON fields
            results["config"] = json.loads(results["config"])
            results["individual_results"] = json.loads(results["individual_results"])
            results["aggregated_metrics"] = json.loads(results["aggregated_metrics"])
            
            return results
            
        except Exception as e:
            print(f"ERROR getting performance results for {job_id}: {e}")
            return None
    
    async def store_regression_analysis(
        self, 
        job_id: str, 
        analysis: Dict[str, Any]
    ) -> bool:
        """Store regression analysis results"""
        
        try:
            await self.connection.execute("""
                INSERT OR REPLACE INTO regression_analysis (
                    job_id, regressions, statistical_analysis, comparison,
                    regression_score, overall_status, analysis_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                job_id,
                json.dumps(analysis.get("regressions", [])),
                json.dumps(analysis.get("statistical_analysis", {})),
                json.dumps(analysis.get("comparison", {})),
                json.dumps(analysis.get("regression_score", {})),
                analysis.get("overall_status", "unknown"),
                analysis.get("analysis_timestamp", datetime.utcnow().isoformat())
            ))
            
            await self.connection.commit()
            return True
            
        except Exception as e:
            print(f"ERROR storing regression analysis for {job_id}: {e}")
            return False
    
    async def get_regression_analysis(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get regression analysis by job ID"""
        
        try:
            cursor = await self.connection.execute("""
                SELECT * FROM regression_analysis WHERE job_id = ?
            """, (job_id,))
            
            row = await cursor.fetchone()
            if not row:
                return None
            
            # Convert row to dictionary
            columns = [description[0] for description in cursor.description]
            analysis = dict(zip(columns, row))
            
            # Parse JSON fields
            analysis["regressions"] = json.loads(analysis["regressions"])
            analysis["statistical_analysis"] = json.loads(analysis["statistical_analysis"])
            analysis["comparison"] = json.loads(analysis["comparison"])
            analysis["regression_score"] = json.loads(analysis["regression_score"])
            
            return analysis
            
        except Exception as e:
            print(f"ERROR getting regression analysis for {job_id}: {e}")
            return None
    
    async def get_baseline_job(self, app_name: str, baseline_version: str) -> Optional[Dict[str, Any]]:
        """Get baseline job for an app version"""
        
        try:
            cursor = await self.connection.execute("""
                SELECT * FROM test_jobs 
                WHERE app_name = ? AND build_version = ? AND submission_type = 'baseline'
                AND status = 'completed'
                ORDER BY created_at DESC
                LIMIT 1
            """, (app_name, baseline_version))
            
            row = await cursor.fetchone()
            if not row:
                return None
            
            # Convert row to dictionary
            columns = [description[0] for description in cursor.description]
            job_data = dict(zip(columns, row))
            
            # Parse JSON fields
            if job_data.get("test_config"):
                job_data["test_config"] = json.loads(job_data["test_config"])
            
            return job_data
            
        except Exception as e:
            print(f"ERROR getting baseline job for {app_name} v{baseline_version}: {e}")
            return None
    
    async def get_app_baselines(self, app_name: str) -> List[Dict[str, Any]]:
        """Get all available baseline versions for an app"""
        
        try:
            cursor = await self.connection.execute("""
                SELECT build_version, created_at, status FROM test_jobs 
                WHERE app_name = ? AND submission_type = 'baseline'
                ORDER BY created_at DESC
            """, (app_name,))
            
            rows = await cursor.fetchall()
            
            baselines = []
            for row in rows:
                baselines.append({
                    "version": row[0],
                    "created_at": row[1],
                    "status": row[2]
                })
            
            return baselines
            
        except Exception as e:
            print(f"ERROR getting baselines for {app_name}: {e}")
            return []
    
    async def get_recent_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent test jobs"""
        
        try:
            cursor = await self.connection.execute("""
                SELECT job_id, app_name, build_version, submission_type, status, created_at
                FROM test_jobs 
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            
            rows = await cursor.fetchall()
            
            jobs = []
            for row in rows:
                jobs.append({
                    "job_id": row[0],
                    "app_name": row[1],
                    "build_version": row[2],
                    "submission_type": row[3],
                    "status": row[4],
                    "created_at": row[5]
                })
            
            return jobs
            
        except Exception as e:
            print(f"ERROR getting recent jobs: {e}")
            return []
    
    async def get_app_performance_history(
        self, 
        app_name: str, 
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get performance history for an app"""
        
        try:
            cursor = await self.connection.execute("""
                SELECT 
                    tj.build_version,
                    tj.created_at,
                    pr.aggregated_metrics
                FROM test_jobs tj
                JOIN performance_results pr ON tj.job_id = pr.job_id
                WHERE tj.app_name = ? AND tj.status = 'completed'
                ORDER BY tj.created_at DESC
                LIMIT ?
            """, (app_name, limit))
            
            rows = await cursor.fetchall()
            
            history = []
            for row in rows:
                metrics = json.loads(row[2]) if row[2] else {}
                history.append({
                    "version": row[0],
                    "timestamp": row[1],
                    "avg_fps": metrics.get("overall_avg_fps", 0),
                    "min_fps": metrics.get("overall_min_fps", 0),
                    "avg_frame_time": metrics.get("overall_avg_frame_time", 0),
                    "comfort_score": metrics.get("overall_comfort_score", 0)
                })
            
            return history
            
        except Exception as e:
            print(f"ERROR getting performance history for {app_name}: {e}")
            return []
    
    async def close(self):
        """Close database connection"""
        if self.connection:
            await self.connection.close()
            print("INFO: Database connection closed") 