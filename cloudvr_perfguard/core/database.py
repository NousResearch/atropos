"""
Database Manager for CloudVR-PerfGuard
Handles storage and retrieval of test jobs, performance data, regression analysis data,
FunSearch evolved functions, and AI Scientist generated papers.
"""

import asyncio
import json
import sqlite3 # Although aiosqlite is used, this import might be present from earlier versions or for type hinting.
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
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
            elif not db_dir: # Handle case where db_path is just a filename in the current directory
                 os.makedirs(".", exist_ok=True)

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
        
        # FunSearch evolved functions table
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS funsearch_evolved_functions (
                function_id TEXT PRIMARY KEY,
                job_id TEXT,
                program_name TEXT NOT NULL,
                evolution_iteration INTEGER NOT NULL,
                evolved_function_code TEXT NOT NULL,
                evaluation_score REAL,
                discovery_timestamp TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (job_id) REFERENCES test_jobs (job_id)
            )
        """)

        # AI Scientist papers table
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS ai_scientist_papers (
                paper_id TEXT PRIMARY KEY,
                job_id TEXT,
                title TEXT NOT NULL,
                abstract TEXT,
                full_text_path TEXT,
                generation_status TEXT DEFAULT 'draft',
                peer_review_feedback TEXT,
                generation_cost REAL,
                publication_details TEXT,
                creation_timestamp TEXT NOT NULL,
                last_updated_timestamp TEXT NOT NULL,
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

    async def store_evolved_function(
        self,
        function_id: str,
        program_name: str,
        evolution_iteration: int,
        evolved_function_code: str,
        discovery_timestamp: str,
        job_id: Optional[str] = None,
        evaluation_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store a FunSearch evolved function."""
        try:
            await self.connection.execute("""
                INSERT INTO funsearch_evolved_functions (
                    function_id, job_id, program_name, evolution_iteration,
                    evolved_function_code, evaluation_score, discovery_timestamp, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                function_id, job_id, program_name, evolution_iteration,
                evolved_function_code, evaluation_score, discovery_timestamp,
                json.dumps(metadata) if metadata else None
            ))
            await self.connection.commit()
            return True
        except Exception as e:
            print(f"ERROR storing evolved function {function_id}: {e}")
            return False

    async def get_evolved_function(self, function_id: str) -> Optional[Dict[str, Any]]:
        """Get an evolved function by its ID."""
        try:
            cursor = await self.connection.execute("""
                SELECT * FROM funsearch_evolved_functions WHERE function_id = ?
            """, (function_id,))
            row = await cursor.fetchone()
            if not row:
                return None
            columns = [description[0] for description in cursor.description]
            func_data = dict(zip(columns, row))
            if func_data.get("metadata"):
                func_data["metadata"] = json.loads(func_data["metadata"])
            return func_data
        except Exception as e:
            print(f"ERROR getting evolved function {function_id}: {e}")
            return None

    async def get_evolved_functions_for_job(self, job_id: str) -> List[Dict[str, Any]]:
        """Get all evolved functions for a given job ID."""
        try:
            cursor = await self.connection.execute("""
                SELECT * FROM funsearch_evolved_functions WHERE job_id = ?
                ORDER BY evolution_iteration
            """, (job_id,))
            rows = await cursor.fetchall()
            functions = []
            for row_data in rows:
                columns = [description[0] for description in cursor.description]
                func_data = dict(zip(columns, row_data))
                if func_data.get("metadata"):
                    func_data["metadata"] = json.loads(func_data["metadata"])
                functions.append(func_data)
            return functions
        except Exception as e:
            print(f"ERROR getting evolved functions for job {job_id}: {e}")
            return []

    async def store_ai_paper(
        self,
        paper_id: str,
        title: str,
        creation_timestamp: str,
        last_updated_timestamp: str,
        job_id: Optional[str] = None,
        abstract: Optional[str] = None,
        full_text_path: Optional[str] = None,
        generation_status: str = 'draft',
        peer_review_feedback: Optional[str] = None,
        generation_cost: Optional[float] = None,
        publication_details: Optional[str] = None
    ) -> bool:
        """Store an AI Scientist generated paper."""
        try:
            await self.connection.execute("""
                INSERT INTO ai_scientist_papers (
                    paper_id, job_id, title, abstract, full_text_path,
                    generation_status, peer_review_feedback, generation_cost,
                    publication_details, creation_timestamp, last_updated_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                paper_id, job_id, title, abstract, full_text_path,
                generation_status, peer_review_feedback, generation_cost,
                publication_details, creation_timestamp, last_updated_timestamp
            ))
            await self.connection.commit()
            return True
        except Exception as e:
            print(f"ERROR storing AI paper {paper_id}: {e}")
            return False

    async def get_ai_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Get an AI paper by its ID."""
        try:
            cursor = await self.connection.execute("""
                SELECT * FROM ai_scientist_papers WHERE paper_id = ?
            """, (paper_id,))
            row = await cursor.fetchone()
            if not row:
                return None
            columns = [description[0] for description in cursor.description]
            paper_data = dict(zip(columns, row))
            return paper_data
        except Exception as e:
            print(f"ERROR getting AI paper {paper_id}: {e}")
            return None

    async def get_ai_papers_for_job(self, job_id: str) -> List[Dict[str, Any]]:
        """Get all AI papers for a given job ID."""
        try:
            cursor = await self.connection.execute("""
                SELECT * FROM ai_scientist_papers WHERE job_id = ?
                ORDER BY creation_timestamp DESC
            """, (job_id,))
            rows = await cursor.fetchall()
            papers = []
            for row_data in rows:
                columns = [description[0] for description in cursor.description]
                papers.append(dict(zip(columns, row_data)))
            return papers
        except Exception as e:
            print(f"ERROR getting AI papers for job {job_id}: {e}")
            return []

    async def update_ai_paper_status(
        self,
        paper_id: str,
        generation_status: str,
        last_updated_timestamp: str,
        abstract: Optional[str] = None,
        full_text_path: Optional[str] = None,
        peer_review_feedback: Optional[str] = None,
        generation_cost: Optional[float] = None,
        publication_details: Optional[str] = None
    ) -> bool:
        """Update status and other fields of an AI paper."""
        try:
            update_fields = {
                "generation_status": generation_status,
                "last_updated_timestamp": last_updated_timestamp
            }
            # Add optional fields to update_fields only if they are provided
            if abstract is not None: update_fields["abstract"] = abstract
            if full_text_path is not None: update_fields["full_text_path"] = full_text_path
            if peer_review_feedback is not None: update_fields["peer_review_feedback"] = peer_review_feedback
            if generation_cost is not None: update_fields["generation_cost"] = generation_cost
            if publication_details is not None: update_fields["publication_details"] = publication_details
            
            set_clause = ", ".join([f"{key} = ?" for key in update_fields.keys()])
            params = list(update_fields.values()) + [paper_id] # paper_id is the last parameter for WHERE clause
            
            await self.connection.execute(f"""
                UPDATE ai_scientist_papers
                SET {set_clause}
                WHERE paper_id = ?
            """, params)
            await self.connection.commit()
            return True
        except Exception as e:
            print(f"ERROR updating AI paper {paper_id}: {e}")
            return False

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
            results_data = dict(zip(columns, row)) # Renamed to avoid conflict
            
            # Parse JSON fields
            results_data["config"] = json.loads(results_data["config"])
            results_data["individual_results"] = json.loads(results_data["individual_results"])
            results_data["aggregated_metrics"] = json.loads(results_data["aggregated_metrics"])
            
            return results_data
            
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
            analysis_data = dict(zip(columns, row)) # Renamed to avoid conflict
            
            # Parse JSON fields
            analysis_data["regressions"] = json.loads(analysis_data["regressions"])
            analysis_data["statistical_analysis"] = json.loads(analysis_data["statistical_analysis"])
            analysis_data["comparison"] = json.loads(analysis_data["comparison"])
            analysis_data["regression_score"] = json.loads(analysis_data["regression_score"])
            
            return analysis_data
            
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
            for row_data in rows: # Renamed to avoid conflict
                baselines.append({
                    "version": row_data[0],
                    "created_at": row_data[1],
                    "status": row_data[2]
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
            for row_data in rows: # Renamed to avoid conflict
                jobs.append({
                    "job_id": row_data[0],
                    "app_name": row_data[1],
                    "build_version": row_data[2],
                    "submission_type": row_data[3],
                    "status": row_data[4],
                    "created_at": row_data[5]
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
            for row_data in rows: # Renamed to avoid conflict
                metrics = json.loads(row_data[2]) if row_data[2] else {}
                history.append({
                    "version": row_data[0],
                    "timestamp": row_data[1],
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