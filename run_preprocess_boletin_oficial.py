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
from pathlib import Path

# Inicializar cliente
CLIENT = genai.Client()
PATH = 'data/raw/sql-console-for-marianbasti-boletin-oficial-argentina.csv'
PATH_POINTS = 'data/processed/boletin-oficial_points.csv'
PATH_LENS = 'data/processed/boletin-oficial_lens.csv'
PATH_DATA = 'data/processed/boletin-oficial_data.csv'
DEBUG = False
BATCH_SIZE = 50  # Reducido para evitar límites de tasa
SLEEP_BETWEEN_BATCHES = 1  # Segundos entre lotes
MAX_RETRIES = 5  # Intentos máximos en caso de error

def get_embedding_batch(texts: List[str], dim: int = 3072, client=CLIENT, retry_count: int = 0) -> np.ndarray:
    """Obtiene embeddings normalizados de Gemini para un lote de textos."""
    try:
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
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
            if retry_count < MAX_RETRIES:
                wait_time = 60 * (retry_count + 1)  # Espera exponencial: 60, 120, 180 segundos
                print(f"         Límite de tasa alcanzado. Esperando {wait_time} segundos...")
                time.sleep(wait_time)
                return get_embedding_batch(texts, dim, client, retry_count + 1)
            else:
                print(f"         Máximo de reintentos alcanzado. Error: {e}")
                raise
        else:
            print(f"         Error inesperado: {e}")
            raise


def get_embedding(texts: List[str], dim: int = 3072, batch_size: int = BATCH_SIZE, client=CLIENT) -> np.ndarray:
    """Obtiene embeddings normalizados de Gemini para similitud semántica.
    Procesa los textos en lotes para evitar límites de payload de la API.
    """
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"      Procesando lote {batch_num}/{total_batches} ({len(batch)} textos)...")
        
        batch_embeddings = get_embedding_batch(batch, dim=dim, client=client)
        all_embeddings.append(batch_embeddings)
        
        # Pausa entre lotes para respetar límites de tasa
        if i + batch_size < len(texts):
            time.sleep(SLEEP_BETWEEN_BATCHES)
    
    return np.vstack(all_embeddings)


def filter_function(texts: List[str], dim: int = 2, batch_size: int = BATCH_SIZE, client=CLIENT) -> np.ndarray:
    """Función de filtro para Mapper: retorna embeddings de baja dimensión."""
    return get_embedding(texts, dim=dim, batch_size=batch_size, client=client)


def save_checkpoint(data_index, embeddings, filepath, dim_name=""):
    """Guarda un checkpoint de embeddings procesados."""
    checkpoint_dir = Path('data/processed/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(embeddings, index=data_index)
    checkpoint_path = checkpoint_dir / f"{Path(filepath).stem}{dim_name}_checkpoint.csv"
    df.to_csv(checkpoint_path)
    print(f"      Checkpoint guardado: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(filepath):
    """Carga un checkpoint si existe."""
    if Path(filepath).exists():
        print(f"      Cargando checkpoint: {filepath}")
        return pd.read_csv(filepath, index_col=0)
    return None


def main():
    print("Iniciando preprocesamiento de datos")
    print(f"Configuración: BATCH_SIZE={BATCH_SIZE}, SLEEP={SLEEP_BETWEEN_BATCHES}s")
    
    print("\n1. Cargando datos...")
    data = pd.read_csv(PATH, index_col='id')
    data = data.dropna(subset=['full_text'])
    data['date'] = pd.to_datetime(data['date'])
    data['new_office'] = (data['date'] >= pd.Timestamp('2023-12-10'))
    print(f"   Cargados {data.shape[0]} registros")
    
    if DEBUG:
        data = data.head(5)
        print(f"   Modo DEBUG: procesando solo {data.shape[0]} registros")

    # Crear directorio de salida
    Path('data/processed').mkdir(parents=True, exist_ok=True)

    print("\n2. Calculando embeddings de alta dimensión (3072 dims)...")
    
    # Intentar cargar checkpoint
    checkpoint_path = Path('data/processed/checkpoints/boletin-oficial_points_checkpoint.csv')
    df_points = load_checkpoint(checkpoint_path)
    
    if df_points is None:
        points = get_embedding(data['full_text'].tolist(), dim=3072, client=CLIENT)
        print(f"   Forma de points: {points.shape}")
        
        # Crear DataFrame para points
        df_points = pd.DataFrame(points, index=data.index)
        print(f"   Forma de df_points: {df_points.shape}")
        
        # Guardar checkpoint
        save_checkpoint(data.index, points, PATH_POINTS)
    else:
        print(f"   ✓ Recuperado desde checkpoint: {df_points.shape}")

    # Guardar archivo final
    df_points.to_csv(PATH_POINTS)
    print(f"   ✓ Guardado df_points en {PATH_POINTS}")
    
    print("\n3. Calculando función de filtro...")
    for dim in [2, 4]:
        print(f"   Dimensión: {dim}")
        
        # Intentar cargar checkpoint
        checkpoint_path = Path(f'data/processed/checkpoints/boletin-oficial_lens_dim{dim}_checkpoint.csv')
        df_lens = load_checkpoint(checkpoint_path)
        
        if df_lens is None:
            lens = filter_function(data['full_text'].tolist(), dim=dim, batch_size=BATCH_SIZE, client=CLIENT)
            print(f"   Forma de lens: {lens.shape}")

            # Crear DataFrame para lens
            df_lens = pd.DataFrame(lens, index=data.index)
            
            # Guardar checkpoint
            save_checkpoint(data.index, lens, PATH_LENS, f"_dim{dim}")
        else:
            print(f"   ✓ Recuperado desde checkpoint: {df_lens.shape}")

        # Guardar archivo final
        df_lens.to_csv(PATH_LENS+f"_dim{dim}.csv")
        print(f"   ✓ Guardado df_lens en {PATH_LENS+f'_dim{dim}.csv'}")

        # Pausa entre dimensiones para evitar límites de tasa
        if dim != 16:  # No pausar después de la última dimensión
            print(f"   Pausa de 10 segundos antes de la siguiente dimensión...")
            time.sleep(10)

    # Guarda data
    data = data[['new_office', 'full_text', 'date']]
    data.to_csv(PATH_DATA)

    print(f"\n✓ Preprocesamiento completo!")
    print(f"✓ Guardado data en {PATH_DATA}")
    print(f"\nResumen:")
    print(f"  - Registros procesados: {data.shape[0]}")
    print(f"  - Dimensiones de df_points: {df_points.shape}")
    print(f"  - Funciones de filtro generadas: 5 (dims: 2, 3, 4, 8, 16)")

if __name__ == "__main__":
    main()
