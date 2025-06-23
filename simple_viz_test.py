"""
Simple test to understand the database structure
"""

import sqlite3

import pandas as pd


def explore_database():
    """Explore the actual database structure"""
    print("🔍 Exploring database structure...")

    conn = sqlite3.connect("results/metrics.db")

    # Check recent data
    print("\n📊 Recent metrics:")
    df = pd.read_sql_query(
        """
        SELECT step, metric_name, value, tags
        FROM metrics
        ORDER BY timestamp DESC
        LIMIT 20
    """,
        conn,
    )

    print(df.to_string())

    # Check steps available
    print(f"\n📈 Steps available: {df['step'].unique()}")
    print(f"📊 Metrics available: {df['metric_name'].unique()}")

    # Check if we have tags
    print(f"\n🏷️ Sample tags: {df['tags'].dropna().head().tolist()}")

    conn.close()


if __name__ == "__main__":
    explore_database()
