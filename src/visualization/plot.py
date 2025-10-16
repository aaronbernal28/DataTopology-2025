import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

def plot_filtration(simplices_maximals, filtration_size, seed=28, show=True):
    '''Plots the filtration of a simplicial complex.
    
    Parameters:
    -----------
    simplices_maximals : dict
        A dictionary where keys are filtration index and values are lists of maximal simplices.
    filtration_size : int
        The number of filtration steps.'''
    np.random.seed(seed)
    points, vertex_ids = calculate_positions(simplices_maximals, filtration_size, seed=seed)

    fig, axs = plt.subplots(1, filtration_size, figsize=(4 * filtration_size, 4))
    
    # Handle case when filtration_size == 1 (axs is not an array)
    if filtration_size == 1:
        axs = [axs]

    for filt_idx, simplices in simplices_maximals.items():
        # Draw all points as background in light gray
        axs[filt_idx].scatter(points[:, 0], points[:, 1], alpha=0.5)

        # Add node identifiers on top of each vertex
        for vertex_id in vertex_ids:
            axs[filt_idx].text(
                points[vertex_id, 0],
                points[vertex_id, 1],
                f"{vertex_id}",
                color='darkblue',
                ha='center',
                va='center',
                fontsize=12,
                zorder=3,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.6)
            )
        
        for simplex in simplices:
            dim = len(simplex)
            if dim == 0:
                axs[filt_idx].scatter(points[simplex[0], 0], points[simplex[0], 1], 
                                     color='blue', zorder=1, s=500)
            elif dim == 1:
                axs[filt_idx].plot(points[simplex, 0], points[simplex, 1], 
                                  color='black', linewidth=2, zorder=2)
            elif dim == 2:
                triangle = plt.Polygon(points[simplex], alpha=0.4, color='green', zorder=1)
                axs[filt_idx].add_patch(triangle)
            elif dim >= 3:
                # For higher dimensional simplices, just show one face
                tetrahedron = plt.Polygon(points[simplex[:3]], alpha=0.4, color='red', zorder=1)
                axs[filt_idx].add_patch(tetrahedron)

        axs[filt_idx].set_title(f'K{filt_idx}')
        axs[filt_idx].set_aspect('equal')
        axs[filt_idx].axis('off')
    
    plt.tight_layout()
    if show:
        plt.show()

def calculate_positions(simplices_maximals, filtration_size, seed=28, iterations=200):
    '''Calculates a fixed position for each vertex in the simplicial complex.
    
    Returns:
    --------
    tuple[np.ndarray, list[int]]
        Array of vertex positions and the list of vertex ids present in the complex.'''
    simplices_maximals_last = simplices_maximals[filtration_size - 1]
    
    # Collect all vertices from all simplices
    all_vertices = set()
    for simplex in simplices_maximals_last:
        all_vertices.update(simplex)
    
    G = nx.Graph()
    G.add_nodes_from(all_vertices)
    
    # Add edges
    for simplex in simplices_maximals_last:
        if len(simplex) >= 2:
            edges = [(simplex[i], simplex[j]) for i in range(len(simplex)) for j in range(i + 1, len(simplex))]
            G.add_edges_from(edges)
    
    pos = nx.spring_layout(G, seed=seed, iterations=iterations)

    max_vertex = max(all_vertices)
    points = np.zeros((max_vertex + 1, 2))
    for vertex_id, (x, y) in pos.items():
        points[vertex_id] = [x, y]

    vertex_ids = sorted(all_vertices)
    return points, vertex_ids