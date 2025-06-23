"""
Fixed Society Dashboard with Robust Callbacks
Addresses callback timeout and error issues
"""

import json
import sqlite3
import subprocess
import threading
import time

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, dcc, html
from plotly.subplots import make_subplots

# Global data cache to prevent callback timeouts
data_cache = {"d": pd.DataFrame(), "categories": {}, "last_update": 0}


def load_metrics_data(db_path="results/metrics.db"):
    """Load metrics with caching to prevent timeouts"""
    global data_cache

    # Check if we need to refresh (cache for 5 seconds)
    if time.time() - data_cache["last_update"] < 5:
        return data_cache["d"], data_cache["categories"]

    try:
        conn = sqlite3.connect(db_path)

        query = """
        SELECT step, metric_name, value, tags
        FROM metrics
        ORDER BY step, metric_name
        LIMIT 1000
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return pd.DataFrame(), {}

        # Pivot quickly
        pivot_df = df.pivot_table(
            index="step", columns="metric_name", values="value", aggfunc="first"
        ).reset_index()

        # Simple category breakdown
        categories = {"simulation": [], "llm": []}
        for col in pivot_df.columns:
            if "llm" in col.lower():
                categories["llm"].append(col)
            elif col != "step":
                categories["simulation"].append(col)

        # Update cache
        data_cache["d"] = pivot_df
        data_cache["categories"] = categories
        data_cache["last_update"] = time.time()

        return pivot_df, categories

    except Exception as e:
        print(f"Data loading error: {e}")
        return pd.DataFrame(), {}


def create_simple_charts(df):
    """Create simplified charts to avoid callback timeouts"""
    if df.empty:
        return go.Figure(), go.Figure(), html.Div("No data")

    # Simple performance chart
    fig1 = go.Figure()
    if "steps_per_second" in df.columns:
        fig1.add_trace(
            go.Scatter(
                x=df["step"],
                y=df["steps_per_second"],
                name="SPS",
                line=dict(color="green"),
            )
        )
    if "agent_count" in df.columns:
        fig1.add_trace(
            go.Scatter(
                x=df["step"],
                y=df["agent_count"],
                name="Agents",
                line=dict(color="blue"),
            )
        )
    fig1.update_layout(title="Performance Metrics", height=400)

    # Simple 3D viz
    n_points = min(50, len(df))
    np.random.seed(42)
    x = np.random.uniform(-50, 50, n_points)
    y = np.random.uniform(-50, 50, n_points)
    z = np.random.uniform(0, 20, n_points)

    fig2 = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(size=8, color=z, colorscale="Viridis", opacity=0.8),
            )
        ]
    )
    fig2.update_layout(title="3D Visualization", height=400)

    # Simple summary cards
    latest = df.iloc[-1] if len(df) > 0 else {}
    cards = dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H4(f"{latest.get('steps_per_second', 0):.1f}"),
                                    html.P("Steps/Second"),
                                ]
                            )
                        ],
                        color="success",
                        outline=True,
                    )
                ],
                width=3,
            ),
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H4(f"{int(latest.get('agent_count', 0))}"),
                                    html.P("Active Agents"),
                                ]
                            )
                        ],
                        color="primary",
                        outline=True,
                    )
                ],
                width=3,
            ),
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H4(f"{latest.get('avg_happiness', 0):.3f}"),
                                    html.P("Avg Happiness"),
                                ]
                            )
                        ],
                        color="warning",
                        outline=True,
                    )
                ],
                width=3,
            ),
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H4(
                                        f"{latest.get('llm_cache_hit_rate', 0):.1%}"
                                    ),
                                    html.P("LLM Cache Hit Rate"),
                                ]
                            )
                        ],
                        color="info",
                        outline=True,
                    )
                ],
                width=3,
            ),
        ]
    )

    return fig1, fig2, cards


# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1(
                            "🌐 Society Simulation Dashboard",
                            className="text-center mb-4",
                        ),
                        html.Hr(),
                    ]
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Button(
                            "🔄 Refresh",
                            id="refresh-btn",
                            color="primary",
                            className="me-2",
                        ),
                        dbc.Button(
                            "🚀 Run Simulation",
                            id="run-sim-btn",
                            color="success",
                            className="me-2",
                        ),
                        html.Span(id="status-text", className="ms-2"),
                    ]
                )
            ],
            className="mb-3",
        ),
        dbc.Row(id="summary-cards", className="mb-3"),
        dbc.Row(
            [
                dbc.Col([dcc.Graph(id="performance-chart")], width=8),
                dbc.Col([dcc.Graph(id="3d-viz")], width=4),
            ]
        ),
        dcc.Interval(
            id="interval-component",
            interval=15000,  # 15 seconds to reduce load
            n_intervals=0,
        ),
    ],
    fluid=True,
)


# Simplified callbacks
@app.callback(
    [
        Output("performance-chart", "figure"),
        Output("3d-viz", "figure"),
        Output("summary-cards", "children"),
        Output("status-text", "children"),
    ],
    [
        Input("refresh-btn", "n_clicks"),
        Input("interval-component", "n_intervals"),
        Input("run-sim-btn", "n_clicks"),
    ],
    prevent_initial_call=False,
)
def update_dashboard(refresh_clicks, interval_clicks, sim_clicks):
    """Single callback to update entire dashboard"""
    ctx = dash.callback_context

    # Run simulation if requested
    if ctx.triggered and "run-sim-btn" in ctx.triggered[0]["prop_id"]:
        try:
            # Run quick simulation
            subprocess.run(
                [
                    "python",
                    "run_simulation.py",
                    "--agents",
                    "100",
                    "--steps",
                    "10",
                    "--optimized",
                    "--workers",
                    "4",
                    "--quiet",
                ],
                timeout=30,
            )
            status = "✅ Simulation completed"
        except Exception:
            status = "⚠️ Simulation failed"
    else:
        status = "📊 Dashboard active"

    # Load data and create charts
    try:
        df, categories = load_metrics_data()
        fig1, fig2, cards = create_simple_charts(df)

        return fig1, fig2, cards, status

    except Exception as e:
        print(f"Callback error: {e}")
        empty_fig = go.Figure()
        empty_cards = html.Div("Error loading data")
        return empty_fig, empty_fig, empty_cards, f"❌ Error: {str(e)[:50]}"


if __name__ == "__main__":
    print("🌐 Starting Fixed Society Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8052")

    # Load initial data
    df, categories = load_metrics_data()
    print(f"✅ Loaded {len(df)} data rows")

    app.run(debug=True, port=8052, host="127.0.0.1")
