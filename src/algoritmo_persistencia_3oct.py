'''
===========================================================================
Fuente:
    Profesor: Gabriel Minian
    Topología aplicada y análisis topológico de datos 2do Cuatrimestre 2025
===========================================================================
'''

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Tuple, Dict

# ============================
# Construicción de la matriz D
# ============================
def build_boundary_matrix(st: 'Filtration') -> Tuple[np.ndarray, List[Tuple[List[int], float]]]:
    ''' Construye la matriz de frontera D a partir de la filtración dada por st (objeto Filtration).
    
    Parameters
    ----------
    st : Filtration
        Objeto que contiene la filtración de un complejo simplicial.
        
    Returns
    -------
    D : np.ndarray
        Matriz de frontera de tamaño N x N, donde N es el número de simplices en la filtración.
        Los elementos son enteros que representan los coeficientes en la frontera.
    simplices : list of tuples
        Lista de tuplas (simplex, filtración) ordenadas por filtración y dimensión.
        Cada tupla contiene una lista de vértices y su valor de filtración.
    '''

    simplices = list(st.get_filtration())
    simplices.sort(key=lambda x: (x[1], len(x[0])))  
    # orden de filtración y después x dimensión (esto asegura que 
    #sigma esté antes que tau si b(sigma)< b(tau) o b(sigma)=b(tau) y sigma es cara de tau)

    N = len(simplices)
    simp_to_idx = {tuple(s[0]): i for i, s in enumerate(simplices)}
    D = np.zeros((N, N), dtype=int)
    
    #hacemos el calculo con coeficientes en Q, por eso ponemos los signos y no reducimos modulo p

    def boundary(simplex: List[int]) -> List[Tuple[Tuple[int, ...], int]]:
        """
        Calcula la frontera de un simplex.
        
        Parameters
        ----------
        simplex : List[int]
            Lista de vértices que forman el simplex.
            
        Returns
        -------
        faces : List[Tuple[Tuple[int, ...], int]]
            Lista de tuplas (cara, signo) donde cada cara es una tupla de vértices
            y signo es ±1 según la orientación.
        """
        faces = []
        for i in range(len(simplex)):
            face = simplex[:i] + simplex[i+1:] #esta es la cara i-esima (que se obtiene sacando el vértice i)
            sign = (-1) ** i
            faces.append((tuple(face), sign))
        return faces

    for j, (sigma, _) in enumerate(simplices):
        sigma = tuple(sigma)
        if len(sigma) > 1:
            for face, sign in boundary(list(sigma)):
                i = simp_to_idx[face]
                D[i, j] = sign

    return D, simplices

# ============================
# Reducción de columnas (cada p tiene que ser el low(q) a lo sumo para una columna q), las columnas con ceros tiene low=-1
# las filas y columnas en python quedan numeradas a partir de 0, así que ponemos low=-1 si son todos ceros y no hay pivote
# ============================
def reduce_matrix(D: np.ndarray) -> np.ndarray:
    """
    Reduce la matriz de frontera D mediante el algoritmo de reducción de columnas.
    
    Parameters
    ----------
    D : np.ndarray
        Matriz de frontera de tamaño N x N a reducir.
        
    Returns
    -------
    Dred : np.ndarray
        Matriz reducida de tamaño N x N donde cada índice de pivote (low) 
        aparece a lo sumo en una columna. Las columnas sin pivote tienen low=-1.
    """
    def low(col: np.ndarray) -> int:
        """
        Encuentra el índice del último elemento no cero en una columna.
        
        Parameters
        ----------
        col : np.ndarray
            Columna de la matriz.
            
        Returns
        -------
        int
            Índice del último elemento no cero, o -1 si la columna es cero.
        """
        rows = np.where(col != 0)[0]
        return rows[-1] if len(rows) else -1

    Dred = D.copy()
    N = D.shape[1]

    for j in range(N):
        while True:
            lows = [low(Dred[:, r]) for r in range(j)]
            lj = low(Dred[:, j])
            if lj in lows and lj != -1:
                r = [r for r in range(j) if low(Dred[:, r]) == lj][0]
                factor = Dred[lj, j] // Dred[lj, r]
                Dred[:, j] -= factor * Dred[:, r]
            else:
                break
    return Dred

# ============================
# Extraer barcodes
# ============================
def extract_barcodes(Dred: np.ndarray, simplices: List[Tuple[List[int], int]]) -> Dict[int, List[Tuple[float, float]]]:
    """
    Extrae los códigos de barra (barcodes) de la homología persistente.
    
    Parameters
    ----------
    Dred : np.ndarray
        Matriz de frontera reducida de tamaño N x N.
    simplices : List[Tuple[List[int], float]]
        Lista de tuplas (simplex, filtración) ordenadas.
        
    Returns
    -------
    barcodes : Dict[int, List[Tuple[float, float]]]
        Diccionario donde las claves son dimensiones y los valores son listas de 
        intervalos (nacimiento, muerte). La muerte puede ser np.inf para barras infinitas.
    """
    def low(col: np.ndarray) -> int:
        """
        Encuentra el índice del último elemento no cero en una columna.
        
        Parameters
        ----------
        col : np.ndarray
            Columna de la matriz.
            
        Returns
        -------
        int
            Índice del último elemento no cero, o -1 si la columna es cero.
        """
        rows = np.where(col != 0)[0]
        return rows[-1] if len(rows) else -1

    barcodes = defaultdict(list)
    N = Dred.shape[1]

    low_of = {}
    for j in range(N):
        lj = low(Dred[:, j])
        if lj != -1:
            low_of[j] = lj

    # Barras finitas
    for j, lj in low_of.items():
        birth_simplex, birth_filtration = simplices[lj]
        death_simplex, death_filtration = simplices[j]
        dim = len(birth_simplex) - 1
        barcodes[dim].append((birth_filtration, death_filtration))

    # Barras infinitas
    pivots = set(low_of.values())
    for j in range(N):
        if low(Dred[:, j]) == -1 and j not in pivots:
            simplex, filt = simplices[j]
            dim = len(simplex) - 1
            barcodes[dim].append((filt, np.inf))

    return barcodes

# ============================
# Gráfico de barcodes 
# ============================
def plot_barcodes(barcodes: Dict[int, List[Tuple[float, float]]], simplices: List[Tuple[List[int], float]]) -> None:
    """
    Grafica los códigos de barra (barcodes) de la homología persistente.
    
    Parameters
    ----------
    barcodes : Dict[int, List[Tuple[float, float]]]
        Diccionario donde las claves son dimensiones y los valores son listas de 
        intervalos (nacimiento, muerte).
    simplices : List[Tuple[List[int], float]]
        Lista de tuplas (simplex, filtración) para determinar el rango de filtración.
        
    Returns
    -------
    None
        Muestra el gráfico usando matplotlib.
    """
    colors = {0: "red", 1: "blue", 2: "green", 3: "purple"}
    fig, ax = plt.subplots(figsize=(8, 5))

    max_filt = int(max(f for _, f in simplices))
    y_offset = 0
    block_gap = 2  # separación extra entre bloques

    for dim, intervals in sorted(barcodes.items()):
        # ordenar barras de más larga a más corta
        intervals_sorted = sorted(intervals, key=lambda x: (max_filt+2 if x[1]==np.inf else x[1]) - x[0], reverse=True)
        for (b, d) in intervals_sorted:
            end = d if d != np.inf else max_filt + 1
            ax.hlines(y_offset, b, end,
                      colors=colors.get(dim, "black"),
                      lw=2)
            y_offset += 1
        # Etiqueta de dimensión al costado izquierdo
        if intervals_sorted:
            mid_y = y_offset - len(intervals_sorted)/2
            ax.text(-0.8, mid_y, f"H{dim}", va="center", ha="right", 
                    fontsize=12, color=colors.get(dim,"black"), weight="bold")
        y_offset += block_gap

       # configurar ticks eje x
    xticks = list(range(0, max_filt+1)) + [max_filt+1]
    xlabels = [str(v) for v in range(0, max_filt+1)] + ["∞"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)

    # sacar eje y
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_ylabel("")

    ax.set_xlabel("Valor filtración")
    ax.set_title("Códigos de barra")

    # sin recuadro de leyenda
    ax.legend().remove() if ax.get_legend() else None


    plt.tight_layout()
    plt.show()

# =======================================
# codigo para crear filtracion sin usar gudhi (si se inserta un simplex y no está alguna de sus caras, se insertan automáticamente al mismo tiempo que el simplex)
# ===========================================

class Filtration:
    """
    Clase para representar una filtración de un complejo simplicial.
    
    Attributes
    ----------
    simplices : Dict[Tuple[int, ...], float]
        Diccionario que mapea cada simplex (como tupla ordenada de vértices) 
        a su valor de filtración.
    """
    def __init__(self) -> None:
        """
        Inicializa una filtración vacía.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        # Guardamos simplices como {tuple(simplex): filtracion}
        self.simplices = {}

    def insert(self, simplex: List[int], filtration: float) -> None:
        """
        Inserta el simplex y todas sus caras si no existen.
        No duplica los ya insertados.
        
        Parameters
        ----------
        simplex : List[int]
            Lista de vértices que forman el simplex a insertar.
        filtration : float
            Valor de filtración en el que aparece el simplex.
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
                self.insert(face, filtration)  # filtración de la cara <= filtración del simplex

    def get_filtration(self) -> List[Tuple[List[int], int]]:
        """
        Retorna la filtración como una lista de tuplas.

        Returns
        -------
        List[Tuple[List[int], int]]
            Lista de tuplas (simplex, filtración), donde cada simplex es una lista de vértices.
        """
        # Retorna lista de (simplex, filtración), convertimos a lista de listas
        return [(list(s), f) for s, f in self.simplices.items()]




# ============================
# EJEMPLO DE USO (abajo se puede cambiar la filtracion por otra)
# ============================
if __name__ == "__main__":
    
    #creamos la filtración usando Filtration(), insertando cada simplex de la filtración
    #al insertar un simplex, inserta automáticamente en ese momento las caras que no estén todavía insertadas

    st = Filtration()

    # Nivel 0 (K0)
    st.insert([0], filtration=0)
    st.insert([1], filtration=0)
    st.insert([2], filtration=0)

    # K1
    st.insert([0,1], filtration=1)
    st.insert([0,2], filtration=1)
    st.insert([1,2], filtration=1)
    st.insert([3], filtration=1)

    # K2
    st.insert([1,3], filtration=2)
    st.insert([2,3], filtration=2)
    
    # K3
    st.insert([1,2,3], filtration=3)

    # K4
    st.insert([1,2,4], filtration=4)
    st.insert([2,3,4], filtration=4)
    st.insert([1,3,4], filtration=4)

    D, simplices = build_boundary_matrix(st)
    Dred = reduce_matrix(D)
    barcodes = extract_barcodes(Dred, simplices)
    print(barcodes)
    plot_barcodes(barcodes, simplices)

