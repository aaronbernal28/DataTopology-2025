import kmapper as km
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA
from typing import List

class BoletinOficialMapper:
    def __init__(self, 
                 points: np.ndarray, # Puntos en el espacio de características
                 lens: np.ndarray = None, # Imagen de los puntos bajo la función de filtro
                 labels: List[str] = None, # Etiquetas de cada documento
                 y: List[int] = None,
                 n_components: int = 2): # Variables objetivo o de coloreo (si las hay)
        self.matrix = cosine_distances(points)
        self.PCA = None
        self.lens = lens
        self.points = points

        if lens is None:
            self.PCA = PCA(n_components=n_components)
            self.lens = self.PCA.fit_transform(self.points)
        
        self.y = y
        self.labels = labels
        self.mapper = km.KeplerMapper(verbose=0)
        self.graph = None
        self.name = None

    def create_mapper_graph(self, 
                            n_cubes: int = 10, 
                            overlap_perc: float = 0.2,
                            clustering_algorithm=None,
                            precomputed_distances: bool = True,
                            args=None):
        """ Crea el grafo de Mapper usando la lente y los puntos proporcionados.
        Parameters:
        ----------
            n_cubes int
                Número de cubos en la cubierta.

            overlap_perc float
                Porcentaje de solapamiento entre cubos.

            clustering_algorithm
                Algoritmo de clustering a usar (por ejemplo, DBSCAN, Kmeans).

            precomputed_distances bool
                Si True, usa la matriz de distancias precomputada.

            args dict: 
                Argumentos adicionales para el algoritmo de clustering.
        """
        if clustering_algorithm is None:
            raise ValueError("clustering_algorithm debe ser proporcionado (por ejemplo, sklearn.cluster.DBSCAN)")
        
        if args is None:
            args = {}
        
        # Configurar el metric según si usamos distancias precomputadas
        cluster_args = args.copy()
        if precomputed_distances:
            cluster_args['metric'] = 'precomputed'
        
        self.graph = self.mapper.map(
            self.lens,
            self.points if not precomputed_distances else self.matrix,
            cover=km.Cover(n_cubes=n_cubes, perc_overlap=overlap_perc),
            clusterer=clustering_algorithm(**cluster_args),
            precomputed=precomputed_distances
        )

        self.name = f"boletin_oficial_mapper_{n_cubes}cubes_{int(overlap_perc*100)}overlap"

    def visualize_mapper_graph(self):
        """Visualiza el grafo de Mapper y lo guarda en un archivo HTML."""
        if self.graph is None:
            raise ValueError("Primero debe crear el grafo usando create_mapper_graph()")
        if self.name is None:
            raise ValueError("No se ha establecido el nombre del grafo")
        
        self.mapper.visualize(
            self.graph,
            path_html= "results/" + self.name + ".html",
            title="Boletín Oficial Mapper Graph",
            color_function=self.y,
            color_function_name='Target Variable'
        )
        self.graph = None
        self.name = None
        self.mapper = km.KeplerMapper(verbose=0)  # Reiniciar KMapper