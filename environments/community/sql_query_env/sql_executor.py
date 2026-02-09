"""
SQL Executor Module

Provides safe SQL execution against in-memory SQLite databases.
Used by the SQL Query Environment for reward verification.
"""

import re
import sqlite3
from typing import Any, List, Optional, Tuple


def create_table_from_wikisql(
    header: List[str],
    rows: List[List[Any]],
    types: Optional[List[str]] = None,
) -> sqlite3.Connection:
    """
    Create an in-memory SQLite table from WikiSQL format.

    Args:
        header: List of column names
        rows: List of row data (each row is a list of values)
        types: Optional list of column types from WikiSQL

    Returns:
        SQLite connection with the table created and populated
    """
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # Use original column names but quoted to handle spaces/special chars
    # SQLite allows any string as identifier when quoted with double quotes
    cols = ", ".join([f'"{h}" TEXT' for h in header])
    cursor.execute(f"CREATE TABLE data ({cols})")

    # Also create the "table" alias since some WikiSQL queries use it
    # Create as a view pointing to data
    try:
        cursor.execute("CREATE VIEW 'table' AS SELECT * FROM data")
    except sqlite3.OperationalError:
        pass  # View already exists or name conflict

    # Insert rows
    if rows:
        placeholders = ", ".join(["?" for _ in header])
        # Convert all values to strings for consistency
        string_rows = [[str(v) if v is not None else "" for v in row] for row in rows]
        cursor.executemany(f"INSERT INTO data VALUES ({placeholders})", string_rows)

    conn.commit()
    return conn


def quote_identifiers_in_sql(sql: str, header: List[str]) -> str:
    """
    Quote column names in SQL that contain special characters.

    WikiSQL's human_readable SQL often has unquoted columns like:
    SELECT State/territory FROM table WHERE ...

    This function adds quotes around column names that need them.
    """
    result = sql
    # Sort by length (longest first) to avoid partial replacements
    sorted_headers = sorted(header, key=len, reverse=True)

    for col in sorted_headers:
        # Skip if already quoted or is a simple identifier
        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", col):
            continue

        # Escape quotes in column name for regex
        col_escaped = re.escape(col)

        # Match the column name when not already quoted
        # Look for the column name not preceded by " and not followed by "
        pattern = rf'(?<!")\b{col_escaped}\b(?!")'
        replacement = f'"{col}"'

        result = re.sub(pattern, replacement, result)

    return result


def execute_sql(conn: sqlite3.Connection, sql: str) -> Optional[List[Tuple]]:
    """
    Execute SQL safely and return results.

    Args:
        conn: SQLite connection
        sql: SQL query to execute

    Returns:
        List of result tuples, or None if execution failed
    """
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        return cursor.fetchall()
    except sqlite3.Error:
        return None
    except Exception:
        return None


def extract_boxed_sql(text: str) -> Optional[str]:
    """
    Extract SQL from \\boxed{} format in LLM response.

    Args:
        text: LLM response text

    Returns:
        Extracted SQL string, or None if not found
    """
    # Try to find \boxed{...} pattern
    # Handle both \\boxed{} and \boxed{} formats
    patterns = [
        r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",  # Handles nested braces
        r"\\boxed\{(.+?)\}",  # Simple pattern
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            sql = match.group(1).strip()
            if sql:
                return sql

    return None


def normalize_result(result: List[Tuple]) -> List[Tuple]:
    """
    Normalize query results for comparison.

    Converts all values to lowercase strings and sorts.
    """
    normalized = []
    for row in result:
        normalized_row = tuple(
            str(v).lower().strip() if v is not None else "" for v in row
        )
        normalized.append(normalized_row)
    return sorted(normalized)


def results_match(result1: List[Tuple], result2: List[Tuple]) -> bool:
    """
    Compare two SQL query results for equality.

    Args:
        result1: First result set
        result2: Second result set

    Returns:
        True if results match (order-independent, case-insensitive)
    """
    if result1 is None or result2 is None:
        return False

    norm1 = normalize_result(result1)
    norm2 = normalize_result(result2)

    return norm1 == norm2
