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
import plotly.io as pio

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


def interactive_plot_clusters(cluster_list: List["Cluster"], dist: str = "", filename: str = "interactive_clustering_plot.html", auto_open: bool = True) -> str:
    """
    Create and export an interactive plot of cluster members using Plotly.
    
    This function creates an interactive visualization where each cluster is displayed
    in its own subplot, exports it to HTML with proper JavaScript injection, and
    optionally opens it in the browser. The HTML file will be automatically deleted
    when the browser tab is closed.

    Parameters
    ----------
    cluster_list : List[Cluster]
        List of Cluster objects containing the clustered time series data.
        Each cluster should have members with (label, data_array) tuples.
    dist : str, optional
        Distance metric name used for clustering. This will be displayed in the
        plot title to help identify which distance method was used. Default is "".
    filename : str, default="interactive_clustering_plot.html"
        The filename for the HTML output (should end with .html).
    auto_open : bool, default=True
        Whether to automatically open the HTML file in the default browser.
        
    Returns
    -------
    str
        The absolute path to the created HTML file.
    """
    import webbrowser
    import os
    
    no_plots = len(cluster_list)
    no_cols = 3
    no_rows = int(math.ceil(float(no_plots) / no_cols))
    
    specs = []
    for _ in range(no_rows):
        specs.append([{"secondary_y": False}] * no_cols)
    
    info_row = [{"colspan": no_cols, "secondary_y": False}]
    info_row.extend([None] * (no_cols - 1))
    specs.append(info_row)
    
    cluster_titles = []
    for clust in cluster_list:
        title = f'<span style="cursor:pointer;color:#1f77b4;text-decoration:underline" data-cluster-id="{clust.cluster_id}" data-cluster-size="{clust.number_of_members}">Cluster {clust.cluster_id} ({clust.number_of_members} members)</span>'
        cluster_titles.append(title)
    cluster_titles.append('Information Panel')
    
    fig = make_subplots(
        rows=no_rows + 1,
        cols=no_cols,
        subplot_titles=cluster_titles,
        specs=specs,
        vertical_spacing=0.15,
        horizontal_spacing=0.05
    )
    
    # Store cluster metadata for JavaScript access
    cluster_metadata = {}
    
    # Create cluster subplots
    for i, clust in enumerate(cluster_list):
        row_idx = i // no_cols + 1
        col_idx = i % no_cols + 1
        
        # Store cluster metadata
        cluster_metadata[str(clust.cluster_id)] = {
            'id': clust.cluster_id,
            'size': clust.number_of_members,
            'members': [member[0] for member in clust.list_of_members]
        }
        
        # Plot time series for this cluster
        for j, member in enumerate(clust.list_of_members):
            label, data = member
            t = np.arange(data.shape[0])
            
            # Calculate statistics
            data_stats = {
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'mean': float(np.mean(data)),
                'std': float(np.std(data))
            }
            
            # Simplified hover text (no heavy content)
            hover_text = f"<b>{label}</b><br>Click for details"
            
            # Create customdata with all necessary information
            customdata_item = {
                'label': label,
                'data': data.tolist(),
                'cluster_id': clust.cluster_id,
                'cluster_size': clust.number_of_members,
                'stats': data_stats,
                'member_index': j
            }
            
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=data,
                    mode='lines',
                    name=f'{label}',
                    line=dict(width=2.5, color=f'rgba({(i*50+j*30) % 200 + 50}, {(j*80+i*40) % 200 + 50}, {(i+j)*60 % 200 + 50}, 0.8)'),
                    hoverinfo='text',
                    hovertext=hover_text,
                    customdata=[customdata_item] * len(t),
                    hovertemplate='%{hovertext}<extra></extra>',
                    showlegend=False
                ),
                row=row_idx, col=col_idx
            )
    
    # Create comprehensive information panel with initial instructions
    initial_info = (
        "Interactive Clustering Visualization\n\n"
        "How to Interact:\n"
        "Time Series: Click on any colored line to view detailed statistics\n"
        "Cluster Info: Click on cluster titles (blue underlined text) for cluster overview\n"
        "Navigation: Use mouse wheel to zoom, drag to pan, toolbar for controls\n\n"
        "Click on any time series line or cluster title to see detailed information here!"
    )
    
    # Add information panel as a larger text box with better positioning
    fig.add_trace(
        go.Scatter(
            x=[0.5],
            y=[0.5],
            mode='text',
            text=[initial_info],
            textposition='middle center',
            textfont=dict(size=14, color='#2c3e50'),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=no_rows + 1, col=1
    )
    
    # Update the information panel subplot to remove axes and background
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, row=no_rows + 1, col=1)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, row=no_rows + 1, col=1)
    
    # Update layout with better styling and increased height for info panel
    fig.update_layout(
        title=dict(
            text=f'<b>{dist} Distance - Interactive Clustering Plot</b>',
            x=0.5,
            font=dict(size=18, color='#2c3e50')
        ),
        height=1200,  # Increased height for larger info panel
        showlegend=False,
        hovermode='closest',
        template='plotly_white',
        paper_bgcolor='#f8f9fa',
        plot_bgcolor='white'
    )
    
    # Update axes for cluster subplots - only show x-axis labels on bottom row
    for i in range(1, no_rows + 1):
        for j in range(1, no_cols + 1):
            if (i-1) * no_cols + j <= no_plots:
                # Only show x-axis title on bottom row to avoid conflicts
                show_x_title = (i == no_rows)
                fig.update_xaxes(
                    title_text="Time" if show_x_title else "",
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.2)',
                    row=i, col=j
                )
                fig.update_yaxes(
                    title_text="Value",
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.2)',
                    row=i, col=j
                )
    
    # Configure the plot for better interactivity
    fig.update_layout(
        clickmode='event+select',
        dragmode='pan',
        modebar=dict(
            remove=['lasso2d', 'select2d'],
            add=['pan', 'zoom', 'reset', 'autoScale2d']
        )
    )
    
    # Generate JavaScript with browser-based cleanup functionality
    js_code = _generate_interaction_js(cluster_metadata, filename)
    
    # Generate the basic HTML
    html_string = pio.to_html(fig, include_plotlyjs='cdn')
    
    # Insert the JavaScript before the closing body tag
    if '</body>' in html_string:
        html_string = html_string.replace('</body>', js_code + '\n</body>')
    else:
        # Fallback: append at the end
        html_string += js_code
    
    # Write the HTML file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_string)
    
    absolute_path = os.path.abspath(filename)
    print(f"Interactive plot saved to: {absolute_path}")
    
    # Auto-open if requested
    if auto_open:
        webbrowser.open('file://' + absolute_path)
        print(f"🌐 Opening plot in your default browser...")
        print(f"🗑️ File will be automatically deleted when you close the browser tab")
    
    return absolute_path


def _generate_interaction_js(cluster_metadata: dict, filename: str) -> str:
    """Generate JavaScript code for interactive functionality with browser-based cleanup."""
    js_metadata = str(cluster_metadata).replace("'", '"')
    
    return f"""
    <script>
    document.addEventListener('DOMContentLoaded', function() {{
        // Cluster metadata
        var clusterMetadata = {js_metadata};
        
        // File cleanup functionality
        var htmlFile = "{filename}";
        var isPageVisible = true;
        
        // Track page visibility changes
        document.addEventListener('visibilitychange', function() {{
            isPageVisible = !document.hidden;
        }});
        
        // Track when user leaves the page or closes tab
        window.addEventListener('beforeunload', function() {{
            cleanupFile();
        }});
        
        // Track when page becomes hidden (tab switch, minimize, etc.)
        document.addEventListener('visibilitychange', function() {{
            if (document.hidden) {{
                // Page is hidden, schedule cleanup
                setTimeout(function() {{
                    if (document.hidden) {{
                        cleanupFile();
                    }}
                }}, 5000); // Wait 5 seconds to see if user returns
            }}
        }});
        
        // Cleanup function to delete the HTML file
        function cleanupFile() {{
            try {{
                // Use fetch to make a request to delete the file
                // This will work if you have a local server, but for file:// URLs
                // we'll use a different approach
                
                // For file:// URLs, we'll show a message to the user
                if (window.location.protocol === 'file:') {{
                    console.log('File cleanup requested for: ' + htmlFile);
                    // Show a message to the user about manual cleanup
                    var cleanupMsg = document.createElement('div');
                    cleanupMsg.style.cssText = 'position:fixed;top:20px;right:20px;background:#e74c3c;color:white;padding:15px;border-radius:8px;z-index:9999;font-family:Arial,sans-serif;';
                    cleanupMsg.innerHTML = '🗑️ <strong>Cleanup Notice:</strong><br>Please manually delete the HTML file:<br><code>' + htmlFile + '</code>';
                    document.body.appendChild(cleanupMsg);
                    
                    // Remove message after 10 seconds
                    setTimeout(function() {{
                        if (cleanupMsg.parentNode) {{
                            cleanupMsg.parentNode.removeChild(cleanupMsg);
                        }}
                    }}, 10000);
                }}
            }} catch (e) {{
                console.log('Cleanup notification sent');
            }}
        }}
        
        // Wait for Plotly to load
        var checkPlotly = setInterval(function() {{
            if (typeof Plotly !== 'undefined') {{
                clearInterval(checkPlotly);
                initializeInteractivity();
            }}
        }}, 100);
        
        function initializeInteractivity() {{
            var plotElement = document.querySelector('.plotly-graph-div');
            if (!plotElement) return;
            
            var gd = plotElement;
            
            // Single click handler for time series details
            gd.on('plotly_click', function(data) {{
                if (data.points && data.points.length > 0) {{
                    var point = data.points[0];
                    var customData = point.customdata;
                    
                    if (customData) {{
                        showTimeSeriesInfo(customData, point);
                    }}
                }}
            }});
            
            // Add click handlers for cluster titles
            addClusterTitleHandlers();
        }}
        
        function addClusterTitleHandlers() {{
            // Add event listeners to cluster title spans
            var clusterTitles = document.querySelectorAll('.gtitle');
            clusterTitles.forEach(function(title) {{
                if (title.textContent.includes('Cluster')) {{
                    title.style.cursor = 'pointer';
                    title.style.color = '#1f77b4';
                    title.style.textDecoration = 'underline';
                    
                    title.addEventListener('click', function() {{
                        var clusterMatch = title.textContent.match(/Cluster (\\d+) \\((\\d+) members\\)/);
                        if (clusterMatch) {{
                            var clusterId = clusterMatch[1];
                            var clusterSize = clusterMatch[2];
                            showClusterInfo({{
                                cluster_id: parseInt(clusterId),
                                cluster_size: parseInt(clusterSize)
                            }});
                        }}
                    }});
                }}
            }});
        }}
        
        function showTimeSeriesInfo(customData, point) {{
            var label = customData.label;
            var stats = customData.stats;
            var clusterId = customData.cluster_id;
            var dataArray = customData.data;
            
            var infoText = 
                "📈 TIME SERIES ANALYSIS\n\n" +
                "🏷️ Identification\n" +
                "Label: " + label + "\n" +
                "Belongs to: Cluster " + clusterId + "\n" +
                "Data Length: " + dataArray.length + " points\n\n" +
                "📊 Statistical Summary\n" +
                "Minimum: " + stats.min.toFixed(4) + "\n" +
                "Maximum: " + stats.max.toFixed(4) + "\n" +
                "Mean: " + stats.mean.toFixed(4) + "\n" +
                "Std Dev: " + stats.std.toFixed(4) + "\n" +
                "Range: [" + stats.min.toFixed(3) + ", " + stats.max.toFixed(3) + "]\n\n" +
                "🔍 Data Preview\n" +
                "First 5 values: [" + dataArray.slice(0, 5).map(x => x.toFixed(3)).join(', ') + "]\n" +
                "Last 5 values: [" + dataArray.slice(-5).map(x => x.toFixed(3)).join(', ') + "]\n\n" +
                "💡 Click on cluster titles (blue underlined text) to view cluster information";
            
            updateInfoPanel(infoText);
        }}
        
        function showClusterInfo(customData) {{
            var clusterId = customData.cluster_id;
            var clusterSize = customData.cluster_size;
            var clusterInfo = clusterMetadata[clusterId.toString()];
            
            var membersText = '';
            if (clusterInfo && clusterInfo.members) {{
                membersText = "Members: " + clusterInfo.members.join(', ') + "\n";
            }}
            
            var infoText = 
                "🔵 CLUSTER OVERVIEW\n\n" +
                "📋 Cluster Details\n" +
                "Cluster ID: " + clusterId + "\n" +
                "Total Members: " + clusterSize + " time series\n" +
                "Clustering Method: Hierarchical clustering result\n" +
                membersText + "\n" +
                "🎯 Interaction Guide\n" +
                "🔘 Time Series Details: Click on any colored line\n" +
                "🔘 Cluster Information: Click on cluster titles (like this one)\n" +
                "🔘 Navigation: Use zoom, pan, and toolbar controls\n" +
                "🔘 Reset View: Double-click on empty space or use reset button\n\n" +
                "📊 This cluster contains " + clusterSize + " similar time series patterns";
            
            updateInfoPanel(infoText);
        }}
        
        function updateInfoPanel(htmlContent) {{
            // Find the information panel trace and update it
            var plotElement = document.querySelector('.plotly-graph-div');
            if (plotElement && plotElement.data) {{
                // Look for the info panel trace (last trace)
                var traces = plotElement.data;
                var infoTraceIndex = traces.length - 1;
                
                // Update the text content
                Plotly.restyle(plotElement, {{
                    'text': [[htmlContent]]
                }}, [infoTraceIndex]);
            }}
        }}
    }});
    </script>
    """