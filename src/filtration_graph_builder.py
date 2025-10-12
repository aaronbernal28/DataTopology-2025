"""
Construcción del grafo del plano proyectivo y extracción de simplices.

Este módulo genera el grafo utilizado para la filtración interactiva,
incluyendo todos los vértices, aristas y triángulos válidos.
"""

import networkx as nx
from typing import List, Dict, Tuple


def construir_grafo_plano_proyectivo():
    """
    Construye el grafo del plano proyectivo con la triangulación estándar.
    
    Returns:
        tuple: (G, VALID_EDGES, POSITIONS, TRIANGLES)
            - G: grafo networkx
            - VALID_EDGES: lista de tuplas (nodo1, nodo2)
            - POSITIONS: diccionario {nodo: (x, y)}
            - TRIANGLES: lista de listas [nodo1, nodo2, nodo3]
    """
    # Crear el grafo con un ciclo exterior de 6 nodos y un ciclo interior de 3 nodos
    ciclo_ext = nx.cycle_graph(6)
    ciclo_int = nx.cycle_graph(3)
    ciclo_int = nx.relabel_nodes(ciclo_int, {i: i + ciclo_ext.number_of_nodes() for i in ciclo_int.nodes})
    G = nx.compose(ciclo_ext, ciclo_int)
    
    # Aristas adicionales para el plano proyectivo
    nuevas_aristas = [
        (0, 7), 
        (1, 7), (1, 8), 
        (2, 8), 
        (3, 6), (3, 8), 
        (4, 6), 
        (5, 6), (5, 7)
    ]
    for src, dst in nuevas_aristas:
        G.add_edge(src, dst)
    
    # Extraer aristas válidas
    VALID_EDGES = list(G.edges())
    
    # Generar posiciones fijas
    POSITIONS = nx.spring_layout(G, seed=3)
    
    # Encontrar todos los triángulos en el grafo
    TRIANGLES = [list(triangle) for triangle in nx.enumerate_all_cliques(G) if len(triangle) == 3]
    
    return G, VALID_EDGES, POSITIONS, TRIANGLES


def nodo_a_id(nodo: int) -> int:
    """
    Convierte un nodo físico a su ID conceptual.
    
    Los nodos 0 y 3 se identifican con ID 1,
    los nodos 1 y 4 se identifican con ID 2,
    los nodos 2 y 5 se identifican con ID 3,
    y los nodos 6, 7, 8 tienen IDs únicos 4, 5, 6.
    
    Args:
        nodo: índice del nodo físico (0-8)
    
    Returns:
        ID conceptual (1-6)
    """
    nodos_a_ids = [1, 2, 3, 1, 2, 3, 4, 5, 6]
    return nodos_a_ids[nodo]


def id_a_nodos(id_conceptual: int) -> List[int]:
    """
    Convierte un ID conceptual a la lista de nodos físicos correspondientes.
    
    Args:
        id_conceptual: ID conceptual (1-6)
    
    Returns:
        Lista de nodos físicos que comparten este ID
    """
    ids_a_nodos = {
        1: [0, 3],
        2: [1, 4], 
        3: [2, 5],
        4: [6], 
        5: [7], 
        6: [8]
    }
    return ids_a_nodos.get(id_conceptual, [])


def arista_a_id_arista(nodo1: int, nodo2: int) -> List[int]:
    """
    Convierte una arista física a una arista conceptual (usando IDs).
    
    Args:
        nodo1: primer nodo físico
        nodo2: segundo nodo físico
    
    Returns:
        Lista ordenada [id1, id2]
    """
    return sorted([nodo_a_id(nodo1), nodo_a_id(nodo2)])


def triangulo_a_id_triangulo(nodo1: int, nodo2: int, nodo3: int) -> List[int]:
    """
    Convierte un triángulo físico a un triángulo conceptual (usando IDs).
    
    Args:
        nodo1, nodo2, nodo3: nodos físicos del triángulo
    
    Returns:
        Lista ordenada [id1, id2, id3]
    """
    return sorted([nodo_a_id(nodo1), nodo_a_id(nodo2), nodo_a_id(nodo3)])


def extraer_todos_los_simplices(G, VALID_EDGES, TRIANGLES) -> Tuple[List, List, List]:
    """
    Extrae todos los simplices válidos del grafo, expresados como IDs conceptuales.
    
    Args:
        G: grafo networkx
        VALID_EDGES: lista de aristas válidas
        TRIANGLES: lista de triángulos válidos
    
    Returns:
        tuple: (vertices_ids, aristas_ids, triangulos_ids)
            - vertices_ids: lista de listas [[1], [2], ..., [6]]
            - aristas_ids: lista de listas [[1, 2], [2, 3], ...]
            - triangulos_ids: lista de listas [[1, 2, 3], ...]
    """
    # Vértices conceptuales (IDs únicos)
    vertices_ids = [[1], [2], [3], [4], [5], [6]]
    
    # Aristas conceptuales (convertir y deduplicar)
    aristas_ids = []
    aristas_vistas = set()
    for nodo1, nodo2 in VALID_EDGES:
        arista_id = tuple(arista_a_id_arista(nodo1, nodo2))
        if arista_id not in aristas_vistas:
            aristas_ids.append(list(arista_id))
            aristas_vistas.add(arista_id)
    
    # Triángulos conceptuales (convertir y deduplicar)
    triangulos_ids = []
    triangulos_vistos = set()
    for triangulo in TRIANGLES:
        triangulo_id = tuple(triangulo_a_id_triangulo(*triangulo))
        if triangulo_id not in triangulos_vistos:
            triangulos_ids.append(list(triangulo_id))
            triangulos_vistos.add(triangulo_id)
    
    return vertices_ids, aristas_ids, triangulos_ids


def obtener_caras(simplex: List[int]) -> List[List[int]]:
    """
    Genera todas las caras propias de un simplex.
    
    Args:
        simplex: lista de IDs [1, 2] o [1, 2, 3]
    
    Returns:
        Lista de caras (cada cara es una lista de IDs)
    """
    if len(simplex) <= 1:
        return []
    
    caras = []
    for i in range(len(simplex)):
        cara = simplex[:i] + simplex[i+1:]
        caras.append(cara)
        # Recursivamente obtener caras de la cara
        caras.extend(obtener_caras(cara))
    
    return caras


def es_cara_de(simplex1: List[int], simplex2: List[int]) -> bool:
    """
    Verifica si simplex1 es una cara de simplex2.
    
    Args:
        simplex1: posible cara
        simplex2: posible cocara
    
    Returns:
        True si simplex1 ⊆ simplex2
    """
    return set(simplex1).issubset(set(simplex2))


if __name__ == "__main__":
    # Prueba del módulo
    G, VALID_EDGES, POSITIONS, TRIANGLES = construir_grafo_plano_proyectivo()
    print(f"Grafo construido:")
    print(f"  Nodos: {G.number_of_nodes()}")
    print(f"  Aristas: {G.number_of_edges()}")
    print(f"  Triángulos: {len(TRIANGLES)}")
    
    vertices_ids, aristas_ids, triangulos_ids = extraer_todos_los_simplices(G, VALID_EDGES, TRIANGLES)
    print(f"\nSimplices conceptuales:")
    print(f"  Vértices: {len(vertices_ids)}")
    print(f"  Aristas: {len(aristas_ids)}")
    print(f"  Triángulos: {len(triangulos_ids)}")
    
    print(f"\nAristas IDs: {aristas_ids}")
    print(f"\nTriángulos IDs: {triangulos_ids}")
