"""
Plotting utilities for clustering results.

This module provides functions for visualizing clustering results including
dendrograms and cluster plots using Plotly for better performance.
"""
import matplotlib.pyplot as plt
import numpy as np
import math
from typing import List, TYPE_CHECKING
from scipy.cluster.hierarchy import dendrogram
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import dash
from dash import dcc, html, Input, Output
import webbrowser
import threading
import time

if TYPE_CHECKING:
    from .clusterer import Cluster

def _plot_dendrogram(z: np.ndarray) -> None:
    """
    Plot hierarchical clustering dendrogram using matplotlib.
    
    This function visualizes the hierarchical clustering structure as a dendrogram,
    showing how clusters are formed at different distances.
    
    Parameters
    ----------
    z : np.ndarray
        Linkage matrix returned by scipy.cluster.hierarchy.linkage.
    """
    dendrogram(z, truncate_mode='lastp', show_leaf_counts=True, show_contracted=True)
    plt.show()


def plot_clusters(cluster_list: List["Cluster"], dist: str, mode: str = 'show', fname: str = 'results') -> None:
    """
    Plot cluster members on separate subplots using matplotlib.

    This function creates a grid of subplots where each subplot shows all time series
    belonging to a specific cluster. Each cluster is displayed in its own subplot
    with all member time series overlaid.

    Parameters
    ----------
    cluster_list : List[Cluster]
        List of Cluster objects containing clustered time series data.
    dist : str
        Distance metric name used for clustering. This will be displayed in the
        window title and can help identify which distance method was used.
    mode : str, default='show'
        Display mode for the plot:
        
        - 'show': Display the plot interactively using matplotlib.pyplot.show()
        - 'save': Save the plot to a PNG file without displaying it
        
    fname : str, default='results'
        Base filename for saving the plot (without extension). Only used when
        mode='save'. The file will be saved as '{fname}.png'.
    """
    main_fig = plt.figure(figsize=(14,10))
    main_fig.canvas.manager.set_window_title(dist + ' distance')
    no_plots = len(cluster_list)
    no_cols = 4
    no_rows = int(math.ceil(float(no_plots) / no_cols))
    i = 1
    
    for clust in cluster_list:
        sub_plot = main_fig.add_subplot(no_rows, no_cols, i)
        i = i + 1

        for j in clust.list_of_members:
            t = np.arange(j[1].shape[0])
            sub_plot.plot(t, j[1], linewidth=2)
        
        plt.title('Cluster no: ' + str(clust.cluster_id), weight='bold')

    plt.tight_layout()
    if mode=='show':
        plt.show()
    elif mode=='save':
        plt.savefig('{0}.png'.format(fname))


def interactive_plot_clusters(cluster_list: List["Cluster"], dist: str, port: int = 8050) -> None:
    """
    Create an enhanced interactive plot of cluster members with optimized layout and auto-browser opening.

    When you click on a time series, information about that time series and its cluster will appear in the information panel.

    Parameters
    ----------
    cluster_list : List[Cluster]
        List of Cluster objects containing clustered time series data.
    dist : str
        Distance metric name used for clustering.
    port : int, default=8050
        Port number for the Dash server.
    """
    
    app = dash.Dash(__name__)
    app.title = f"Interactive Clustering - {dist}"
    
    no_plots = len(cluster_list)
    no_cols = 3
    cluster_rows = int(math.ceil(float(no_plots) / no_cols))
    
    colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2 + px.colors.qualitative.Set3
    
    cluster_fig = make_subplots(
        rows=cluster_rows, 
        cols=no_cols,
        subplot_titles=[f"Cluster {clust.cluster_id}" for clust in cluster_list],
        vertical_spacing=0.12,
        horizontal_spacing=0.04
    )
    
    cluster_data = {}
    time_series_data = {}
    
    for idx, clust in enumerate(cluster_list):
        row = (idx // no_cols) + 1
        col = (idx % no_cols) + 1

        cluster_data[clust.cluster_id] = {
            'cluster_id': clust.cluster_id,
            'number_of_members': clust.number_of_members,
            'indices_of_members': clust.indices_of_members.tolist(),
            'best_representative_member': clust.best_representative_member[0],
            'members': [member[0] for member in clust.list_of_members]
        }
        
        for j_idx, (name, ts_data) in enumerate(clust.list_of_members):
            t = np.arange(ts_data.shape[0])
            ts_id = f"cluster_{clust.cluster_id}_ts_{j_idx}"
            
            time_series_data[ts_id] = {
                'name': name,
                'cluster_id': clust.cluster_id,
                'length': len(ts_data),
                'mean': float(np.mean(ts_data)),
                'std': float(np.std(ts_data)),
                'min': float(np.min(ts_data)),
                'max': float(np.max(ts_data)),
                'is_representative': name == clust.best_representative_member[0],
                'data': ts_data.tolist()
            }
            
            is_repr = name == clust.best_representative_member[0]
            line_color = '#e74c3c' if is_repr else colors[j_idx % len(colors)]
            line_width = 3.5 if is_repr else 2.5
            opacity = 1.0 if is_repr else 0.8
            
            cluster_fig.add_trace(
                go.Scatter(
                    x=t,
                    y=ts_data,
                    mode='lines',
                    name=name,
                    line=dict(width=line_width, color=line_color),
                    opacity=opacity,
                    customdata=[ts_id] * len(t),
                    hovertemplate=f"<b>{name}</b><br>" +
                                "Time: %{x}<br>" +
                                "Value: %{y:.3f}<br>" +
                                f"Cluster: {clust.cluster_id}<br>" +
                                f"{'Representative' if is_repr else ''}<br>" +
                                "<extra></extra>",
                    showlegend=False
                ),
                row=row, col=col
            )
    
    cluster_fig.update_layout(
        height=280 * cluster_rows,
        showlegend=False,
        clickmode='event+select',
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='#f8f9fa',
        font={'family': 'Arial, sans-serif'}
    )
    
    cluster_fig.update_annotations(font_size=18, font_color='#1a202c', font=dict(weight='bold'))
    
    info_panel_height = 350
    
    app.layout = html.Div([
        html.Div([
            html.H1([
                f"Interactive Clustering - {dist} Distance"
            ], style={
                'textAlign': 'center', 
                'marginBottom': '4px',
                'color': '#1a202c',
                'fontFamily': 'Arial, sans-serif'
            })
        ]),
        
        html.Div([
            dcc.Graph(
                id='cluster-plot',
                figure=cluster_fig,
                style={'marginBottom': '2px'}
            )
        ], style={'marginTop': '0px'}),
        
        html.Div([
            html.Div([
                html.Div(id='info-panel', children=[])
            ], style={
                'backgroundColor': '#ffffff',
                'border': '1px solid #e2e8f0',
                'borderRadius': '16px',
                'padding': '18px',
                'margin': '4px auto',
                'width': 'calc(100% - 160px)',
                'boxShadow': '0 10px 25px rgba(0, 0, 0, 0.08)',
                'minHeight': f'{info_panel_height}px',
                'maxHeight': 'none',
                'overflow': 'visible',
                'boxSizing': 'border-box',
                'resize': 'both',
                'overflowY': 'visible'
            })
        ])
    ], style={
        'backgroundColor': '#f8f9fa',
        'minHeight': '100vh',
        'padding': '15px',
        'fontFamily': 'Arial, sans-serif'
    })
    
    @app.callback(
        Output('info-panel', 'children'),
        [Input('cluster-plot', 'clickData')]
    )
    def update_info_panel(clickData):
        if clickData is None:
            return [
                html.H2([
                    "Information Panel"
                ], style={'textAlign': 'center', 'marginBottom': '20px', 'color': '#1a202c'}),
                html.Div([
                    html.Div([
                        html.P([
                            html.B("Representative members"), " are highlighted in ", 
                            html.Span("red", style={'color': '#e74c3c', 'fontWeight': 'bold'})
                        ], style={'textAlign': 'center', 'fontSize': '16px'}),
                        html.P("Click on any time series line to see detailed statistics", 
                               style={'textAlign': 'center', 'color': '#7f8c8d'})
                    ], style={
                        'padding': '36px',
                        'backgroundColor': '#ffffff',
                        'borderRadius': '12px',
                        'textAlign': 'center',
                        'border': '1px solid #e2e8f0'
                    })
                ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'height': '100%'})
            ]
        
        try:
            point = clickData['points'][0]
            customdata = point.get('customdata')
            
            if customdata and customdata in time_series_data:
                ts_info = time_series_data[customdata]
                cluster_info = cluster_data[ts_info['cluster_id']]
                
                return [html.Div([
                    html.Div([
                        # Left column
                        html.Div([
                            html.Div([
                                html.Span("Time Series Information", style={'fontWeight': 'bold', 'color': '#34495e', 'fontSize': '16px'})
                            ], style={'marginBottom': '15px', 'overflow': 'hidden'}),
                            html.Div([
                                html.Span("Label:", style={'fontWeight': 'bold', 'color': '#34495e', 'minWidth': '60px', 'display': 'inline-block'}),
                                html.Span(f" {ts_info['name']}", style={'marginLeft': '10px', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'})
                            ], style={'marginBottom': '10px', 'overflow': 'hidden'}),
                            html.Div([
                                html.Span("Length:", style={'fontWeight': 'bold', 'color': '#34495e', 'minWidth': '60px', 'display': 'inline-block'}),
                                html.Span(f" {ts_info['length']}", style={'marginLeft': '10px', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'})
                            ], style={'marginBottom': '10px', 'overflow': 'hidden'}),
                            html.Div([
                                html.Span("Mean:", style={'fontWeight': 'bold', 'color': '#34495e', 'minWidth': '60px', 'display': 'inline-block'}),
                                html.Span(f" {ts_info['mean']:.4f}", style={'marginLeft': '10px', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'})
                            ], style={'marginBottom': '10px', 'overflow': 'hidden'})
                        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'overflow': 'hidden'}),
                        
                        # Middle column
                        html.Div([
                            html.Div([
                                html.Span("Std Dev:", style={'fontWeight': 'bold', 'color': '#34495e', 'minWidth': '80px', 'display': 'inline-block'}),
                                html.Span(f" {ts_info['std']:.4f}", style={'marginLeft': '10px', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'})
                            ], style={'marginBottom': '10px', 'overflow': 'hidden'}),
                            html.Div([
                                html.Span("Max Value:", style={'fontWeight': 'bold', 'color': '#34495e', 'minWidth': '80px', 'display': 'inline-block'}),
                                html.Span(f" {ts_info['max']:.4f}", style={'marginLeft': '10px', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'})
                            ], style={'marginBottom': '10px', 'overflow': 'hidden'}),
                            html.Div([
                                html.Span("Min Value:", style={'fontWeight': 'bold', 'color': '#34495e', 'minWidth': '80px', 'display': 'inline-block'}),
                                html.Span(f" {ts_info['min']:.4f}", style={'marginLeft': '10px', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'})
                            ], style={'marginBottom': '10px', 'overflow': 'hidden'}),
                            html.Div([
                                html.Span("Representative:", style={'fontWeight': 'bold', 'color': '#34495e', 'minWidth': '80px', 'display': 'inline-block'}),
                                html.Span(" Yes" if ts_info["is_representative"] else " No", 
                                         style={'marginLeft': '10px', 'color': '#34495e' if ts_info["is_representative"] else '#34495e', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'})
                            ], style={'marginBottom': '10px', 'overflow': 'hidden'})
                        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%', 'overflow': 'hidden'}),
                        
                        # Gray vertical line
                        html.Div(style={'width': '1px', 'backgroundColor': '#cccccc', 'height': '120px', 'display': 'inline-block', 'marginLeft': '2%', 'marginRight': '2%', 'verticalAlign': 'top'}),
                        
                        # Right column
                        html.Div([
                            html.Div([
                                html.Span("Cluster Information", style={'fontWeight': 'bold', 'color': '#34495e', 'fontSize': '16px'})
                            ], style={'marginBottom': '15px', 'overflow': 'hidden'}),
                            html.Div([
                                html.Span("Cluster ID:", style={'fontWeight': 'bold', 'color': '#34495e', 'minWidth': '100px', 'display': 'inline-block'}),
                                html.Span(f" {ts_info['cluster_id']}", style={'marginLeft': '10px', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'})
                            ], style={'marginBottom': '10px', 'overflow': 'hidden'}),
                            html.Div([
                                html.Span("Cluster Size:", style={'fontWeight': 'bold', 'color': '#34495e', 'minWidth': '100px', 'display': 'inline-block'}),
                                html.Span(f" {cluster_info['number_of_members']}", style={'marginLeft': '10px', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'})
                            ], style={'marginBottom': '10px', 'overflow': 'hidden'}),
                            html.Div([
                                html.Span("Representative Member:", style={'fontWeight': 'bold', 'color': '#34495e', 'minWidth': '100px', 'display': 'inline-block'}),
                                html.Span(f" {cluster_info['best_representative_member']}", style={'marginLeft': '10px', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'})
                            ], style={'marginBottom': '10px', 'overflow': 'hidden'})
                        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'overflow': 'hidden'})
                    ], style={'wordWrap': 'break-word', 'borderBottom': '2px solid #ecf0f1', 'paddingBottom': '12px', 'marginBottom': '16px'}),
                    
                    # Cluster members section
                    html.Div([
                        html.H4("Cluster Members:", style={'color': '#34495e', 'marginTop': '16px', 'marginBottom': '8px'}),
                        html.Div([
                            html.Span(
                                member + ("" if member == cluster_info['best_representative_member'] else ""), 
                                style={
                                    'display': 'inline-block',
                                    'margin': '2px 6px',
                                    'padding': '4px 8px',
                                    'backgroundColor': '#e74c3c' if member == cluster_info['best_representative_member'] else '#3498db',
                                    'color': 'white',
                                    'borderRadius': '12px',
                                    'fontSize': '12px',
                                    'maxWidth': '200px',
                                    'overflow': 'hidden',
                                    'textOverflow': 'ellipsis',
                                    'whiteSpace': 'nowrap'
                                }
                            ) for member in cluster_info['members']
                        ], style={
                            'wordWrap': 'break-word',
                            'overflowWrap': 'break-word',
                            'maxWidth': '100%'
                        })
                    ])
                ], style={
                    'width': '100%',
                    'maxWidth': '100%',
                    'wordWrap': 'break-word',
                    'boxSizing': 'border-box'
                })]
        except (KeyError, IndexError, TypeError):
            return [
                html.Div([
                    html.P("Error: Unable to display information for the selected time series.", 
                           style={'textAlign': 'center', 'color': '#e74c3c', 'fontSize': '16px'})
                ], style={'padding': '20px'})
            ]

    app.cluster_data = cluster_data
    app.time_series_data = time_series_data
    
    def open_browser():
        time.sleep(1.5)  # Wait for server to start
        webbrowser.open(f'http://127.0.0.1:{port}')

    threading.Thread(target=open_browser, daemon=True).start()

    try:
        app.run(debug=False, port=port, host='127.0.0.1', 
                dev_tools_silence_routes_logging=True, 
                use_reloader=False)
    except Exception as e:
        print(f"Error starting server: {e}")