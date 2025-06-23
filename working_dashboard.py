"""
Working Society Metrics Dashboard
Uses actual simulation data structure
"""

import json
import sqlite3
import subprocess
import threading
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from plotly.subplots import make_subplots


class WorkingSocietyDashboard:
    def __init__(self, db_path="results/metrics.db"):
        self.db_path = db_path
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.setup_layout()
        self.setup_callbacks()

    def load_metrics_data(self):
        """Load metrics from the actual database structure"""
        try:
            conn = sqlite3.connect(self.db_path)

            # Load all metrics
            query = """
            SELECT step, metric_name, value, tags
            FROM metrics
            ORDER BY step, metric_name
            """

            df = pd.read_sql_query(query, conn)
            conn.close()

            if df.empty:
                return pd.DataFrame(), {}

            # Pivot to get metrics as columns
            pivot_df = df.pivot_table(
                index="step", columns="metric_name", values="value", aggfunc="first"
            ).reset_index()

            # Get category breakdown
            category_breakdown = {}
            for _, row in df.iterrows():
                try:
                    tags = json.loads(row["tags"]) if row["tags"] else {}
                    category = tags.get("category", "unknown")
                    if category not in category_breakdown:
                        category_breakdown[category] = []
                    if row["metric_name"] not in category_breakdown[category]:
                        category_breakdown[category].append(row["metric_name"])
                except Exception:
                    pass

            return pivot_df, category_breakdown

        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame(), {}

    def create_performance_chart(self, df):
        """Create performance metrics chart"""
        if df.empty:
            return go.Figure()

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Performance",
                "Agent Metrics",
                "LLM Usage",
                "System Health",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": True}, {"secondary_y": False}],
            ],
        )

        steps = df["step"]

        # Performance metrics
        if "steps_per_second" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=df["steps_per_second"],
                    name="SPS",
                    line=dict(color="green"),
                ),
                row=1,
                col=1,
            )
        if "runtime" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=steps, y=df["runtime"], name="Runtime", line=dict(color="blue")
                ),
                row=1,
                col=1,
            )

        # Agent metrics
        if "agent_count" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=steps, y=df["agent_count"], name="Agents", line=dict(color="red")
                ),
                row=1,
                col=2,
            )
        if "avg_energy" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=df["avg_energy"],
                    name="Avg Energy",
                    line=dict(color="orange"),
                ),
                row=1,
                col=2,
            )
        if "avg_happiness" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=df["avg_happiness"],
                    name="Avg Happiness",
                    line=dict(color="purple"),
                ),
                row=1,
                col=2,
            )

        # LLM metrics
        if "llm_total_requests" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=df["llm_total_requests"],
                    name="LLM Requests",
                    line=dict(color="cyan"),
                ),
                row=2,
                col=1,
            )
        if "llm_cache_hit_rate" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=df["llm_cache_hit_rate"],
                    name="Cache Hit Rate",
                    line=dict(color="magenta"),
                ),
                row=2,
                col=1,
                secondary_y=True,
            )

        # System health
        if "total_social_interactions" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=df["total_social_interactions"],
                    name="Interactions",
                    line=dict(color="brown"),
                ),
                row=2,
                col=2,
            )
        if "total_objects_created" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=df["total_objects_created"],
                    name="Objects Created",
                    line=dict(color="pink"),
                ),
                row=2,
                col=2,
            )

        fig.update_layout(
            height=700, showlegend=True, title_text="Society Simulation Metrics"
        )
        return fig

    def create_summary_cards(self, df):
        """Create summary cards with latest metrics"""
        if df.empty:
            return html.Div("No data available")

        latest = df.iloc[-1] if len(df) > 0 else {}

        cards = []

        # Performance card
        if "steps_per_second" in latest:
            cards.append(
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H4(
                                            f"{latest['steps_per_second']:.1f}",
                                            className="card-title",
                                        ),
                                        html.P("Steps/Second", className="card-text"),
                                    ]
                                )
                            ],
                            color="success",
                            outline=True,
                        )
                    ],
                    width=3,
                )
            )

        # Agent count card
        if "agent_count" in latest:
            cards.append(
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H4(
                                            f"{int(latest['agent_count'])}",
                                            className="card-title",
                                        ),
                                        html.P("Active Agents", className="card-text"),
                                    ]
                                )
                            ],
                            color="primary",
                            outline=True,
                        )
                    ],
                    width=3,
                )
            )

        # Happiness card
        if "avg_happiness" in latest:
            cards.append(
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H4(
                                            f"{latest['avg_happiness']:.3f}",
                                            className="card-title",
                                        ),
                                        html.P("Avg Happiness", className="card-text"),
                                    ]
                                )
                            ],
                            color="warning",
                            outline=True,
                        )
                    ],
                    width=3,
                )
            )

        # LLM efficiency card
        if "llm_cache_hit_rate" in latest:
            cards.append(
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H4(
                                            f"{latest['llm_cache_hit_rate']:.1%}",
                                            className="card-title",
                                        ),
                                        html.P(
                                            "LLM Cache Hit Rate", className="card-text"
                                        ),
                                    ]
                                )
                            ],
                            color="info",
                            outline=True,
                        )
                    ],
                    width=3,
                )
            )

        return dbc.Row(cards) if cards else html.Div("No metrics available")

    def create_3d_visualization(self, df):
        """Create 3D visualization of society state"""
        if df.empty or len(df) < 2:
            return go.Figure()

        # Create synthetic 3D data for visualization
        n_points = min(len(df), 100)
        np.random.seed(42)

        x = np.random.uniform(-50, 50, n_points)
        y = np.random.uniform(-50, 50, n_points)
        z = np.random.uniform(0, 20, n_points)

        # Use happiness as color if available
        if "avg_happiness" in df.columns:
            colors = np.repeat(df["avg_happiness"].iloc[-1], n_points)
            color_title = "Avg Happiness"
        elif "avg_energy" in df.columns:
            colors = np.repeat(df["avg_energy"].iloc[-1], n_points)
            color_title = "Avg Energy"
        else:
            colors = z
            color_title = "Z Position"

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=colors,
                        colorscale="Viridis",
                        opacity=0.8,
                        colorbar=dict(title=color_title),
                    ),
                    text=[
                        f"Society Node {i}<br>{color_title}: {c:.3f}"
                        for i, c in enumerate(colors)
                    ],
                    hovertemplate="%{text}<extra></extra>",
                )
            ]
        )

        fig.update_layout(
            title="3D Society Visualization",
            scene=dict(
                xaxis_title="Social Space X",
                yaxis_title="Social Space Y",
                zaxis_title="Activity Level",
            ),
            width=500,
            height=400,
        )

        return fig

    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = dbc.Container(
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
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Controls"),
                                        dbc.CardBody(
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
                                                html.Div(
                                                    id="status-div", className="mt-2"
                                                ),
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=12,
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
                    interval=10000,  # Update every 10 seconds
                    n_intervals=0,
                ),
                dcc.Store(id="metrics-store"),
                dcc.Store(id="categories-store"),
            ],
            fluid=True,
        )

    def setup_callbacks(self):
        """Setup dashboard callbacks"""

        @self.app.callback(
            [
                Output("metrics-store", "data"),
                Output("categories-store", "data"),
                Output("status-div", "children"),
            ],
            [
                Input("refresh-btn", "n_clicks"),
                Input("interval-component", "n_intervals"),
                Input("run-sim-btn", "n_clicks"),
            ],
            prevent_initial_call=False,
        )
        def update_data(refresh_clicks, interval_clicks, sim_clicks):
            import dash

            ctx = dash.callback_context

            status = dbc.Alert("Data loaded", color="success", is_open=True, fade=True)

            # Run simulation if requested
            if ctx.triggered and "run-sim-btn" in ctx.triggered[0]["prop_id"]:
                status = dbc.Alert(
                    "🚀 Running simulation...", color="info", is_open=True
                )
                threading.Thread(target=self.run_simulation).start()

            # Load data
            df, categories = self.load_metrics_data()

            return df.to_dict("records"), categories, status

        @self.app.callback(
            Output("summary-cards", "children"),
            Input("metrics-store", "data"),
            prevent_initial_call=False,
        )
        def update_summary(data):
            if not data:
                return html.Div("No data available")
            df = pd.DataFrame(data)
            return self.create_summary_cards(df)

        @self.app.callback(
            Output("performance-chart", "figure"),
            Input("metrics-store", "data"),
            prevent_initial_call=False,
        )
        def update_performance_chart(data):
            if not data:
                return go.Figure()
            df = pd.DataFrame(data)
            return self.create_performance_chart(df)

        @self.app.callback(
            Output("3d-viz", "figure"),
            Input("metrics-store", "data"),
            prevent_initial_call=False,
        )
        def update_3d_viz(data):
            if not data:
                return go.Figure()
            df = pd.DataFrame(data)
            return self.create_3d_visualization(df)

    def run_simulation(self):
        """Run simulation in background"""
        try:
            subprocess.run(
                [
                    "python",
                    "run_simulation.py",
                    "--agents",
                    "150",
                    "--steps",
                    "20",
                    "--optimized",
                    "--workers",
                    "4",
                    "--quiet",
                ],
                check=True,
            )
        except Exception as e:
            print(f"Simulation error: {e}")

    def run(self, debug=False, port=8051):
        """Run the dashboard"""
        print("🌐 Starting Working Society Dashboard...")
        print(f"📊 Dashboard available at: http://localhost:{port}")
        print(f"💾 Using database: {self.db_path}")

        try:
            # Test data loading first
            df, categories = self.load_metrics_data()
            print(
                f"✅ Loaded {len(df)} data rows with categories: {list(categories.keys())}"
            )

            self.app.run(debug=debug, port=port, host="127.0.0.1")
        except Exception as e:
            print(f"❌ Dashboard startup error: {e}")
            import traceback

            traceback.print_exc()


def main():
    dashboard = WorkingSocietyDashboard()
    dashboard.run(debug=True)


if __name__ == "__main__":
    main()
