"""
Script de preprocesamiento para datos del Boletín Oficial Argentina.
Crea embeddings y funciones de filtro para análisis con Mapper.
Salidas: df_points y df_lens guardados en data/processed/
"""

import time
import numpy as np
import pandas as pd
import google.genai.types as types
import google.genai as genai
from typing import List

# Inicializar cliente
CLIENT = genai.Client()
PATH = 'data/raw/sql-console-for-marianbasti-boletin-oficial-argentina.csv'
PATH_POINTS = 'data/processed/boletin-oficial_points.csv'
PATH_LENS = 'data/processed/boletin-oficial_lens.csv'
PATH_DATA = 'data/processed/boletin-oficial_data.csv'
DEBUG = True

def get_embedding(texts: List[str], dim: int = 3072, client=CLIENT) -> np.ndarray:
    """Obtiene embeddings normalizados de Gemini para similitud semántica."""
    result = [
        e.values for e in client.models.embed_content(
            model="gemini-embedding-001",
            contents=texts,
            config=types.EmbedContentConfig(
                output_dimensionality=dim,
                task_type="SEMANTIC_SIMILARITY"
            )).embeddings
    ]
    result = np.array(result)  ## (N, dim)
    norms = np.linalg.norm(result, axis=1, keepdims=True)  ## (N, 1)
    return result / norms  ## (N, dim)


def filter_function(texts: List[str], dim: int = 2, client=CLIENT) -> np.ndarray:
    """Función de filtro para Mapper: retorna embeddings de baja dimensión."""
    return get_embedding(texts, dim=dim, client=client)


def main():
    print("Iniciando preprocesamiento de datos")
    
    print("\n1. Cargando datos...")
    data = pd.read_csv(PATH, index_col='id')
    data = data.dropna(subset=['full_text'])
    data['date'] = pd.to_datetime(data['date'])
    data['new_office'] = (data['date'] >= pd.Timestamp('2023-12-10'))
    print(f"   Cargados {data.shape[0]} registros")
    
    if DEBUG:
        data = data.head(5)
        print(f"   Modo DEBUG: procesando solo {data.shape[0]} registros")

    print("\n2. Calculando embeddings de alta dimensión (3072 dims)...")
    points = get_embedding(data['full_text'].tolist(), dim=3072, client=CLIENT)
    print(f"   Forma de points: {points.shape}")
    
    # Crear DataFrame para points
    df_points = pd.DataFrame(points, index=data.index)
    print(f"   Forma de df_points: {df_points.shape}")

    df_points.to_csv(PATH_POINTS)
    print(f"   ✓ Guardado df_points en {PATH_POINTS}")
    
    print("\n3. Calculando función de filtro...")
    for dim in [2, 3, 4, 8, 16]:
        print(f"Dimensión: {dim}")

        lens = filter_function(data['full_text'].tolist(), dim=dim, client=CLIENT)
        print(f"   Forma de lens: {lens.shape}")

        # Crear DataFrame para lens
        df_lens = pd.DataFrame(lens, index=data.index)

        # Guarda df_lens
        df_lens.to_csv(PATH_LENS+f"_dim{dim}.csv")
        print(f"   ✓ Guardado df_lens en {PATH_LENS+f'_dim{dim}.csv'}")

        time.sleep(10)  # Pausa para evitar límites de tasa

    # Guarda data
    data = data[['new_office', 'full_text', 'date']]
    data.to_csv(PATH_DATA)
    print(f"✓ Guardado data en {PATH_DATA}")

if __name__ == "__main__":
    main()
