"""
Advanced 3D Visualization Dashboard for Society Simulation
Optimized for memory efficiency and real-time performance
"""

import json
import sqlite3
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import datetime
import subprocess
import time
import threading
from collections import deque

class SocietyVisualizationDashboard:
    def __init__(self, db_path="results/metrics.db", max_points=1000):
        self.db_path = db_path
        self.max_points = max_points  # Limit points for memory efficiency
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.data_buffer = deque(maxlen=max_points)
        self.setup_layout()
        self.setup_callbacks()
        
    def load_simulation_data(self):
        """Load simulation data with memory limits"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Load metrics data and pivot to get agent data
            query = """
            SELECT step, metric_name, value, tags
            FROM metrics 
            ORDER BY step DESC, timestamp DESC
            LIMIT ?
            """
            
            df = pd.read_sql_query(query, conn, params=(self.max_points * 10,))
            conn.close()
            
            if df.empty:
                return pd.DataFrame()
                
            # Parse tags to get agent_id when available
            df['agent_id'] = df['tags'].str.extract(r'agent_id:(\d+)')
            df['agent_id'] = pd.to_numeric(df['agent_id'], errors='coerce')
            
            # Pivot to get metrics as columns
            pivot_df = df.pivot_table(
                index=['step', 'agent_id'], 
                columns='metric_name', 
                values='value', 
                aggfunc='first'
            ).reset_index()
            
            # Ensure we have the basic columns we need
            required_cols = ['step']
            for col in required_cols:
                if col not in pivot_df.columns:
                    pivot_df[col] = 0
                    
            # Fill missing agent positions with random values for visualization
            if 'agent_id' in pivot_df.columns and pivot_df['agent_id'].notna().any():
                # Generate mock positions if not in data
                n_agents = len(pivot_df.dropna(subset=['agent_id']))
                if n_agents > 0:
                    np.random.seed(42)
                    if 'x' not in pivot_df.columns:
                        pivot_df['x'] = np.random.uniform(-50, 50, len(pivot_df))
                    if 'y' not in pivot_df.columns:
                        pivot_df['y'] = np.random.uniform(-50, 50, len(pivot_df))
                    if 'z' not in pivot_df.columns:
                        pivot_df['z'] = np.random.uniform(0, 10, len(pivot_df))
            
            return pivot_df.fillna(0)
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def create_3d_scatter(self, df, step=None):
        """Create optimized 3D scatter plot"""
        if df.empty:
            return go.Figure()
            
        # Filter to specific step or latest
        if step is not None:
            plot_df = df[df['step'] == step]
        else:
            latest_step = df['step'].max()
            plot_df = df[df['step'] == latest_step]
        
        # Subsample if too many points
        if len(plot_df) > 500:
            plot_df = plot_df.sample(n=500)
        
        # Get color column (use first available numeric column)
        numeric_cols = plot_df.select_dtypes(include=[np.number]).columns
        color_col = None
        for col in ['happiness', 'energy', 'wealth']:
            if col in numeric_cols:
                color_col = col
                break
        if not color_col and len(numeric_cols) > 2:
            color_col = numeric_cols[2]  # Use third numeric column if available
        
        # Create hover text
        hover_text = []
        for _, row in plot_df.iterrows():
            text_parts = []
            if 'agent_id' in plot_df.columns and pd.notna(row['agent_id']):
                text_parts.append(f"Agent {int(row['agent_id'])}")
            for col in ['energy', 'wealth', 'happiness']:
                if col in plot_df.columns and pd.notna(row[col]):
                    text_parts.append(f"{col.title()}: {row[col]:.1f}")
            hover_text.append("<br>".join(text_parts) if text_parts else "Agent")
        
        fig = go.Figure(data=[go.Scatter3d(
            x=plot_df['x'],
            y=plot_df['y'], 
            z=plot_df['z'],
            mode='markers',
            marker=dict(
                size=5,
                color=plot_df[color_col] if color_col else plot_df['x'],
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title=color_col.title() if color_col else "Position")
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>'
        )])
        
        fig.update_layout(
            title="3D Agent Distribution",
            scene=dict(
                xaxis_title="X Position",
                yaxis_title="Y Position", 
                zaxis_title="Z Position"
            ),
            width=600,
            height=500
        )
        
        return fig
    
    def create_metrics_timeline(self, df):
        """Create timeline of key metrics"""
        if df.empty:
            return go.Figure()
            
        # Get available numeric columns for metrics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        metric_cols = [col for col in numeric_cols if col not in ['step', 'agent_id']]
        
        if not metric_cols:
            return go.Figure()
            
        # Aggregate by step for available metrics
        agg_dict = {}
        for col in metric_cols:
            if col in df.columns:
                agg_dict[col] = ['mean', 'std']
        
        if not agg_dict:
            return go.Figure()
            
        metrics = df.groupby('step').agg(agg_dict).round(2)
        
        # Create subplots based on available metrics
        n_metrics = len(metric_cols)
        rows = (n_metrics + 1) // 2
        cols = 2 if n_metrics > 1 else 1
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=metric_cols[:4],  # Limit to 4 for display
            specs=[[{"secondary_y": True} for _ in range(cols)] for _ in range(rows)]
        )
        
        steps = metrics.index
        colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan']
        
        # Plot available metrics
        for i, col in enumerate(metric_cols[:4]):  # Limit to 4 subplots
            row = (i // 2) + 1
            col_pos = (i % 2) + 1
            color = colors[i % len(colors)]
            
            if (col, 'mean') in metrics.columns:
                fig.add_trace(go.Scatter(x=steps, y=metrics[(col, 'mean')], 
                                        name=f'Avg {col}', line=dict(color=color)), 
                             row=row, col=col_pos)
            
            if (col, 'std') in metrics.columns:
                fig.add_trace(go.Scatter(x=steps, y=metrics[(col, 'std')], 
                                        name=f'{col} Std', line=dict(color=color, dash='dash')), 
                             row=row, col=col_pos, secondary_y=True)
        
        fig.update_layout(height=600, showlegend=False, title_text="Society Metrics Over Time")
        return fig
    
    def create_agent_heatmap(self, df):
        """Create 2D density heatmap"""
        if df.empty:
            return go.Figure()
            
        latest_step = df['step'].max()
        plot_df = df[df['step'] == latest_step]
        
        # Create 2D histogram
        fig = go.Figure(data=go.Histogram2d(
            x=plot_df['x'],
            y=plot_df['y'],
            colorscale='Blues',
            nbinsx=20,
            nbinsy=20
        ))
        
        fig.update_layout(
            title="Agent Density Map",
            xaxis_title="X Position",
            yaxis_title="Y Position",
            width=500,
            height=400
        )
        
        return fig
    
    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("🌐 Society Simulation Dashboard", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Real-time Controls"),
                        dbc.CardBody([
                            dbc.Button("🔄 Refresh Data", id="refresh-btn", color="primary", className="me-2"),
                            dbc.Button("▶️ Run New Simulation", id="run-sim-btn", color="success", className="me-2"),
                            html.Div(id="status-div", className="mt-2")
                        ])
                    ])
                ], width=12)
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="3d-scatter")
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="density-heatmap")
                ], width=6)
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="metrics-timeline")
                ], width=12)
            ]),
            
            dcc.Interval(
                id='interval-component',
                interval=5000,  # Update every 5 seconds
                n_intervals=0
            ),
            
            dcc.Store(id='data-store')
            
        ], fluid=True)
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @callback(
            [Output('data-store', 'data'),
             Output('status-div', 'children')],
            [Input('refresh-btn', 'n_clicks'),
             Input('interval-component', 'n_intervals'),
             Input('run-sim-btn', 'n_clicks')]
        )
        def update_data(refresh_clicks, interval_clicks, sim_clicks):
            ctx = dash.callback_context
            
            status = dbc.Alert("Data loaded successfully", color="success", is_open=True, fade=True)
            
            # Run new simulation if requested
            if ctx.triggered and 'run-sim-btn' in ctx.triggered[0]['prop_id']:
                try:
                    status = dbc.Alert("🚀 Running new simulation...", color="info", is_open=True)
                    # Run simulation in background (non-blocking)
                    threading.Thread(target=self.run_background_simulation).start()
                except Exception as e:
                    status = dbc.Alert(f"Error: {e}", color="danger", is_open=True)
            
            # Load current data
            df = self.load_simulation_data()
            return df.to_dict('records'), status
        
        @callback(
            Output('3d-scatter', 'figure'),
            Input('data-store', 'data')
        )
        def update_3d_scatter(data):
            df = pd.DataFrame(data)
            return self.create_3d_scatter(df)
        
        @callback(
            Output('density-heatmap', 'figure'),
            Input('data-store', 'data')
        )
        def update_heatmap(data):
            df = pd.DataFrame(data)
            return self.create_agent_heatmap(df)
        
        @callback(
            Output('metrics-timeline', 'figure'),
            Input('data-store', 'data')
        )
        def update_timeline(data):
            df = pd.DataFrame(data)
            return self.create_metrics_timeline(df)
    
    def run_background_simulation(self):
        """Run simulation in background"""
        try:
            subprocess.run([
                "python", "run_simulation.py", 
                "--agents", "200", "--steps", "25", 
                "--optimized", "--workers", "4", "--quiet"
            ], check=True)
        except Exception as e:
            print(f"Simulation error: {e}")
    
    def run(self, debug=False, port=8050):
        """Run the dashboard"""
        print(f"🌐 Starting Society Visualization Dashboard...")
        print(f"📊 Dashboard will be available at: http://localhost:{port}")
        print(f"💾 Using database: {self.db_path}")
        
        self.app.run_server(debug=debug, port=port, host='0.0.0.0')

def main():
    """Main function to run the dashboard"""
    dashboard = SocietyVisualizationDashboard()
    dashboard.run(debug=True)

if __name__ == "__main__":
    main()