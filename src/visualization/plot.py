import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

def faces(simplex):
    '''Generates all faces of a simplex'''
    n = len(simplex)
    res = []
    for i in range(1 << n):
        face = list(simplex[j] for j in range(n) if (i & (1 << j)))
        res.append(face)
    return res[1:]

def plot_filtration(simplices_maximals, filtration_size, seed=28, show=True):
    '''Plots the filtration of a simplicial complex.
    
    Parameters:
    -----------
    simplices_maximals : dict
        A dictionary where keys are filtration index and values are lists of maximal simplices.
    filtration_size : int
        The number of filtration steps.'''
    np.random.seed(seed)
    points = calculate_positions(simplices_maximals, filtration_size, seed=seed)

    fig, axs = plt.subplots(1, filtration_size, figsize=(4 * filtration_size, 4))
    
    # Handle case when filtration_size == 1 (axs is not an array)
    if filtration_size == 1:
        axs = [axs]

    for filt_idx, simplices in simplices_maximals.items():
        # Draw all points as background in light gray

        for simplex in simplices:
            for subsimplex in faces(simplex):
                dim = len(subsimplex) - 1
                if dim == 0:
                    axs[filt_idx].scatter(points[subsimplex[0], 0], points[subsimplex[0], 1], 
                                         color='lightblue', zorder=1, s=400, alpha=0.8)
                    axs[filt_idx].text(
                        points[subsimplex[0], 0],
                        points[subsimplex[0], 1],
                        f"{subsimplex[0]}",
                        color='darkblue',
                        ha='center',
                        va='center',
                        fontsize=12,
                        zorder=filtration_size,
                        bbox=dict(boxstyle='round,pad=0.15', fc='lightblue', ec='none', alpha=0.6)
                    )

                elif dim == 1:
                    axs[filt_idx].plot(points[subsimplex, 0], points[subsimplex, 1], 
                                      color='black', linewidth=2, zorder=2)
                elif dim == 2:
                    triangle = plt.Polygon(points[subsimplex], alpha=0.6, color='lightblue', zorder=2)
                    axs[filt_idx].add_patch(triangle)
                elif dim >= 3:
                    # For higher dimensional simplices, just show one face
                    tetrahedron = plt.Polygon(points[subsimplex[:3]], alpha=0.6, color='red', zorder=2)
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

    return points