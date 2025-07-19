import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import networkx as nx
from matplotlib.axes import Axes
try:
    from reportlab.pdfgen import canvas
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
from typing import Optional
from matplotlib.animation import FuncAnimation
import numpy as np

def plot_tradeoff(df: pd.DataFrame, x: str, y: str, color: str, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    scatter = ax.scatter(df[x], df[y], c=df[color], cmap='coolwarm', s=100, alpha=0.7)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f'Trade-off: {x} vs. {y} (color={color})')
    plt.colorbar(scatter, ax=ax)
    return ax

def export_csv(df: pd.DataFrame, filename: str):
    df.to_csv(filename, index=False)

def export_json(df: pd.DataFrame, filename: str):
    df.to_json(filename, orient='records')

def export_pdf(df: pd.DataFrame, filename: str):
    if not PDF_AVAILABLE:
        raise ImportError('reportlab is not installed')
    c = canvas.Canvas(filename)
    c.drawString(100, 800, 'AI Constitution Simulator Report')
    y = 780
    for col in df.columns:
        c.drawString(100, y, f'{col}: {df[col].tolist()}')
        y -= 20
        if y < 100:
            c.showPage()
            y = 800
    c.save()

def plot_agent_timeline(agent_log: list, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    years = list(range(len(agent_log)))
    ax.plot(years, agent_log)
    ax.set_xlabel('Year')
    ax.set_ylabel('Value')
    ax.set_title('Agent Timeline')
    return ax

def animate_network_evolution(history, model, filename: str = "network_evolution.gif"):
    """Animate the evolution of the social network and ideology over time."""
    G = nx.Graph()
    for c in model.citizens:
        G.add_node(c.id, type='citizen', ideology=c.ideology)
        for f in c.social_links:
            G.add_edge(c.id, f)
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(8, 6))
    node_list = list(G.nodes)
    cmap = plt.get_cmap('coolwarm')

    def update(frame):
        ax.clear()
        # Use ideology from history if available
        if 'ideology' in history:
            node_colors = np.array([history['ideology'][frame][n] if n in history['ideology'][frame] else 0.0 for n in node_list], dtype=float)
        else:
            node_colors = np.zeros(len(node_list))
        nodes = nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_color=node_colors, cmap=cmap, ax=ax, node_size=100, vmin=-1, vmax=1)  # type: ignore
        edges = nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)
        ax.set_title(f'Social Network (Year {frame+1})')
        ax.axis('off')
        return [nodes] + list(edges)

    anim = FuncAnimation(fig, update, frames=len(history['year']), interval=500)
    anim.save(filename, writer='pillow')
    plt.close(fig)
    return filename

def detect_communities(model) -> dict:
    """Detect communities in the agent social network using networkx."""
    G = nx.Graph()
    for c in model.citizens:
        G.add_node(c.id, type='citizen', ideology=c.ideology)
        for f in c.social_links:
            G.add_edge(c.id, f)
    try:
        from networkx.algorithms import community
        communities = list(community.greedy_modularity_communities(G))
        return {i: list(comm) for i, comm in enumerate(communities)}
    except ImportError:
        return {}

def plot_communities(model, ax: Optional[Axes] = None):
    """Plot detected communities in the social network."""
    G = nx.Graph()
    for c in model.citizens:
        G.add_node(c.id, type='citizen', ideology=c.ideology)
        for f in c.social_links:
            G.add_edge(c.id, f)
    communities = detect_communities(model)
    cmap = plt.get_cmap('tab10')
    pos = nx.spring_layout(G, seed=42)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    for i, comm in communities.items():
        color = np.array([cmap(i % 10)])
        nx.draw_networkx_nodes(G, pos, nodelist=comm, node_color=color, ax=ax, node_size=100)  # type: ignore
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)
    ax.set_title('Agent Communities')
    ax.axis('off')
    return ax

def generate_pdf_report(model, metrics_df, filename: str = "simulation_report.pdf"):
    """Generate a styled PDF report with summary, plots, and key findings."""
    if not PDF_AVAILABLE:
        raise ImportError('reportlab is not installed')
    c = canvas.Canvas(filename)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 800, 'AI Constitution Simulator Report')
    c.setFont("Helvetica", 12)
    y = 780
    c.drawString(100, y, f"Years simulated: {model.years}")
    y -= 20
    c.drawString(100, y, f"Final Gini: {metrics_df['gini'].iloc[-1]:.3f}")
    y -= 20
    c.drawString(100, y, f"Final Avg Dissent: {metrics_df['avg_dissent'].iloc[-1]:.3f}")
    y -= 20
    c.drawString(100, y, f"Final Avg Happiness: {metrics_df['avg_happiness'].iloc[-1]:.3f}")
    y -= 20
    c.drawString(100, y, f"Final Polarization: {metrics_df['polarization'].iloc[-1]:.3f}")
    y -= 20
    c.drawString(100, y, f"Final Corruption: {metrics_df['avg_corruption'].iloc[-1]:.3f}")
    y -= 20
    c.drawString(100, y, f"Final Press Freedom: {metrics_df['press_freedom'].iloc[-1]:.3f}")
    y -= 20
    c.drawString(100, y, f"Total Protests: {metrics_df['protests'].sum()}")
    y -= 20
    c.drawString(100, y, f"Laws Overturned: {metrics_df['laws_overturned'].iloc[-1]}")
    y -= 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, y, "Key Findings:")
    c.setFont("Helvetica", 12)
    y -= 20
    findings = [
        f"Inequality trended {'up' if metrics_df['gini'].iloc[-1] > metrics_df['gini'].iloc[0] else 'down'}.",
        f"Unrest {'increased' if metrics_df['avg_dissent'].iloc[-1] > metrics_df['avg_dissent'].iloc[0] else 'decreased'}."
    ]
    for finding in findings:
        c.drawString(120, y, finding)
        y -= 20
    c.save()
    return filename 