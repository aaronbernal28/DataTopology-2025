import numpy as np
import gudhi
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from typing import List, Tuple, Dict

def plot_persistence_barcode(persistence: List[Tuple[int, Tuple[float, float]]], ax = None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    x_vals = sorted([death for _, (_, death) in persistence if death != float('inf')]
                    + [birth for _, (birth, _) in persistence])

    bar_idx = 0
    for dim in sorted(set(d for d, _ in persistence)):
        intervals = [interval for d, interval in persistence if d == dim]
        for i, (birth, death) in enumerate(intervals):
            if death == float('inf'):
                death = max(x_vals) + 1
            ax.plot([birth, death], [bar_idx, bar_idx], marker='', c=f'C{dim}', linewidth=2, label=f'Dim {dim}' if i==0  else "")
            # Markers at birth and death
            ax.plot(birth, bar_idx, marker='|', c=f'C{dim}')
            ax.plot(death, bar_idx, marker='4', c=f'C{dim}', markersize=5)
            bar_idx += 1

    ax.set_xlabel('Filtration Value')
    ax.set_ylabel('Barcode Index')
    ax.set_title('Barcode')
    ax.set_yticks([])
    ax.set_xlim(right=max(x_vals) + 1.5)
    x_labels = [f'{x:.2f}' for x in x_vals] + ['∞']
    ax.set_xticks(x_vals + [max(x_vals) + 1], x_labels)
    if len(persistence) < 20:
        ax.set_ylim(-10/len(persistence), len(persistence)*2)
    ax.legend()

def plot_persistence(persistence: List[Tuple[int, Tuple[float, float]]], ax = None):
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

    gudhi.plot_persistence_diagram(persistence, axes=ax[0])
    plot_persistence_barcode(persistence, ax=ax[1])

    ax[0].set_title("Persistence Diagram")
    ax[1].set_title("Barcode")

    plt.tight_layout()
    plt.show()

def plot_triangles(triangles: List[Tuple[int, int, int]], ax, pos: Dict[int, np.ndarray[float, float]]):
    for triangle in triangles:
        triangle_coords = [pos[node] for node in triangle]
        triangle_patch = Polygon(triangle_coords, fill=True, facecolor='gray', alpha=0.3)
        ax.add_patch(triangle_patch)


class Filtration_plano_proyectivo:
    def __init__(self, valid_edges: List[Tuple[int, int]], pos: Dict[int, Tuple[float, float]]):
        # Guardamos simplices como {tuple(simplex): filtracion}
        self.simplices = {}
        self.id_to_nodes = {1: [0, 3],
                            2: [1, 4], 
                            3: [2, 5],
                            4: [6], 5: [7], 6: [8]}
        self.node_to_id = [1, 2, 3, 1, 2, 3, 4, 5, 6]
        self.labels = {node: f'{id}' for node, id in enumerate(self.node_to_id)}
        self.valid_edges = valid_edges
        self.pos = pos

    def insert(self, simplex: List[int], filtration: int) -> None:
        """
        Inserta el simplex y todas sus caras si no existen.
        No duplica los ya insertados.
        """
        simplex = tuple(sorted(simplex))  # ordenamos para consistencia

        # Si ya existe, actualizar filtración solo si es menor
        if simplex in self.simplices:
            self.simplices[simplex] = min(self.simplices[simplex], filtration)
        else:
            self.simplices[simplex] = filtration

        # Insertar recursivamente todas las caras
        if len(simplex) > 1:
            for i in range(len(simplex)):
                face = simplex[:i] + simplex[i+1:]
                self.insert(list(face), filtration)  # filtración de la cara <= filtración del simplex

    def get_filtration(self) -> List[Tuple[List[int], int]]:
        # Retorna lista de (simplex, filtración), convertimos a lista de listas
        return [(list(s), f) for s, f in self.simplices.items()]

    def persistence(self, Dred: np.ndarray, simplices):
        """
        Parametros:
        ----------
        Dred: matriz de frontera reducida
        simplices_list: output of build_boundary_matrix(st)
        
        Returns:
        -------
        Retorna lista de tuplas (dimensión, (nacimiento, muerte))
        """
        def low(col):
            rows = np.where(col != 0)[0]
            return rows[-1] if len(rows) else -1

        barcodes = []
        N = Dred.shape[1]

        low_of = {}
        for j in range(N):
            # 
            lj = low(Dred[:, j])
            if lj != -1:
                low_of[j] = lj

        # Barras finitas
        for j, lj in low_of.items():
            birth_simplex, birth_filtration = simplices[lj]
            death_simplex, death_filtration = simplices[j]
            dim = len(birth_simplex) - 1
            barcodes.append((dim, (birth_filtration, death_filtration)))

        # Barras infinitas
        pivots = set(low_of.values())
        for j in range(N):
            if low(Dred[:, j]) == -1 and j not in pivots:
                simplex, filt = simplices[j]
                dim = len(simplex) - 1
                barcodes.append((dim, (filt, np.inf)))
                
        # Remove birth >= death cases
        barcodes = sorted([b for b in barcodes if b[1][0] < b[1][1]], key=lambda x: (x[1][0], x[1][1]), reverse=True)
        return barcodes
    
    def plot_triangles(self, triangles, ax):
        '''
        triangles: list of tuples (id1, id2, id3)
            But each id corresponds to multiple nodes in the graph
        '''
        for (idx, idy, idz) in triangles:
            for nodex in self.id_to_nodes[idx]:
                for nodey in self.id_to_nodes[idy]:
                    for nodez in self.id_to_nodes[idz]:
                        # Check if all three edges exist (in either direction)
                        edge_xy = (nodex, nodey) in self.valid_edges or (nodey, nodex) in self.valid_edges
                        edge_yz = (nodey, nodez) in self.valid_edges or (nodez, nodey) in self.valid_edges
                        edge_zx = (nodez, nodex) in self.valid_edges or (nodex, nodez) in self.valid_edges
                        
                        if edge_xy and edge_yz and edge_zx:
                            triangle_coords = [self.pos[nodex], self.pos[nodey], self.pos[nodez]]
                            triangle_patch = Polygon(triangle_coords, fill=True, facecolor='gray', alpha=0.3)
                            ax.add_patch(triangle_patch)

    def plot_filtration(self, filtration, ax):
        K_i = nx.empty_graph(self.pos.keys())
        edges = []
        triangles = []
        for s, f in self.simplices.items():
            if f <= filtration:
                if len(s) == 1:
                    # Add all actual nodes corresponding to this conceptual ID
                    conceptual_id = s[0]
                    for actual_node in self.id_to_nodes[conceptual_id]:
                        K_i.add_node(actual_node)
                elif len(s) == 2:
                    edges.append(s)
                else:
                    triangles.append(s)

        # Add edges to the graph based on conceptual IDs
        for id1, id2 in edges:
            for n1 in self.id_to_nodes[id1]:
                for n2 in self.id_to_nodes[id2]:
                    if (n1, n2) in self.valid_edges or (n2, n1) in self.valid_edges:
                        K_i.add_edge(n1, n2)
        
        nx.draw(K_i, self.pos, labels=self.labels, with_labels=True, node_color='lightblue', ax=ax)
        self.plot_triangles(triangles, ax=ax)

    def plot(self, filtration_size=None):
        if filtration_size is None:
            filtration_size = int(max(self.simplices.values())) + 1

        plt.subplots(1, filtration_size, figsize=(3*filtration_size, 3))

        for i in range(filtration_size):
            plt.subplot(1, filtration_size, i+1)
            self.plot_filtration(i, plt.gca())
            plt.title(f'K_{i}')
            plt.tight_layout()
        plt.show()

    def distance_matriz(self) -> np.ndarray:
        ''' Genera una matriz de distancias ficticia usando get_filtration().
        Tiene valores enteros no negativos.'''
        dist = np.full((6, 6), np.inf)
        
        # Diagonal
        for i in range(6):
            dist[i, i] = 0
        
        # Cada (2-simplex) tiene distancia filtration
        for simplex, filtration in self.simplices.items():
            if len(simplex) == 2:
                id1, id2 = simplex
                i, j = id1 - 1, id2 - 1
                dist[i, j] = filtration
                dist[j, i] = filtration

        return dist