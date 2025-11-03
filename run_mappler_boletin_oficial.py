import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from src.mapper import BoletinOficialMapper

# Cargar datos preprocesados
df = pd.read_csv('data/processed/boletin-oficial_data.csv', index_col='id')
df_points = pd.read_csv('data/processed/boletin-oficial_points.csv', index_col='id')
df_lens2 = pd.read_csv('data/processed/boletin-oficial_lens.csv_dim2.csv', index_col='id')
df_lens4 = pd.read_csv('data/processed/boletin-oficial_lens.csv_dim4.csv', index_col='id')

data = {}
data['points'] = df_points.values
data['lens2'] = df_lens2.values
data['lens4'] = df_lens4.values
data['labels'] = None
data['y'] = df['new_office'].astype(int).tolist()

# Crear instancia del mapper
mapper = BoletinOficialMapper(
    points=data['points'], 
    lens=data['lens2'], 
    labels=data['labels'],
    y=data['y']
    )

# Crear el grafo de Mapper
mapper.create_mapper_graph(
    n_cubes=30, 
    overlap_perc=0.3,
    clustering_algorithm=DBSCAN,
    precomputed_distances=True,
    args={'eps': 0.3, 'min_samples': 3}
    )

# Visualizar el grafo de Mapper
mapper.visualize_mapper_graph()
print("Mapper graph visualizado y guardado exitosamente!")