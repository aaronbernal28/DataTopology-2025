"""
Script de prueba para verificar la instalación y funcionamiento básico
de los módulos de filtración interactiva.
"""

import sys
import os

# Agregar el directorio raíz al path
directorio_raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, directorio_raiz)

def test_filtration_graph_builder():
    """Prueba el módulo de construcción del grafo."""
    print("=" * 60)
    print("Test 1: Módulo filtration_graph_builder")
    print("=" * 60)
    
    from src.filtration_graph_builder import (
        construir_grafo_plano_proyectivo,
        extraer_todos_los_simplices,
        nodo_a_id,
        id_a_nodos
    )
    
    # Construir grafo
    G, VALID_EDGES, POSITIONS, TRIANGLES = construir_grafo_plano_proyectivo()
    print(f"✓ Grafo construido: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")
    
    # Extraer simplices
    vertices, aristas, triangulos = extraer_todos_los_simplices(G, VALID_EDGES, TRIANGLES)
    print(f"✓ Simplices extraídos: {len(vertices)} vértices, {len(aristas)} aristas, {len(triangulos)} triángulos")
    
    # Probar conversiones
    assert nodo_a_id(0) == 1, "Conversión nodo 0 -> ID 1 fallida"
    assert nodo_a_id(3) == 1, "Conversión nodo 3 -> ID 1 fallida"
    assert id_a_nodos(1) == [0, 3], "Conversión ID 1 -> [0, 3] fallida"
    print(f"✓ Conversiones nodo ↔ ID funcionan correctamente")
    
    print("✓ Test 1 completado exitosamente\n")
    return True


def test_interactive_filtration():
    """Prueba el módulo de filtración interactiva."""
    print("=" * 60)
    print("Test 2: Módulo interactive_filtration_dash")
    print("=" * 60)
    
    try:
        from src.interactive_filtration_dash import ConstructorFiltracionInteractivo
        
        # Crear instancia
        constructor = ConstructorFiltracionInteractivo(tamano_filtracion=6)
        print(f"✓ Constructor creado con tamaño de filtración {constructor.tamano_filtracion}")
        
        # Verificar estado inicial
        estado = constructor.estado_inicial
        print(f"✓ Estado inicial creado con {len(estado['simplices'])} pasos")
        
        # Probar activación de simplex
        estado_test = estado.copy()
        estado_test['simplices'] = {i: [] for i in range(6)}
        estado_test['agregados_en_paso'] = {i: [] for i in range(6)}
        
        # Activar un vértice
        estado_test = constructor.activar_simplex([1], 0, estado_test)
        assert [1] in estado_test['simplices'][0], "Activación de vértice fallida"
        assert [1] in estado_test['simplices'][1], "Propagación de vértice fallida"
        print(f"✓ Activación y propagación de simplices funciona")
        
        # Probar figura
        fig = constructor.crear_figura_filtracion(estado)
        print(f"✓ Figura de filtración creada con {len(fig.data)} trazos")
        
        print("✓ Test 2 completado exitosamente\n")
        return True
        
    except ImportError as e:
        print(f"⚠ Advertencia: No se pudo importar Dash (esperado si no está instalado)")
        print(f"  Error: {e}")
        print("  Instale con: pip install dash plotly jupyter-dash")
        print("✓ Test 2 omitido (dependencias opcionales no instaladas)\n")
        return True


def test_integration_with_existing_code():
    """Prueba la integración con el código existente."""
    print("=" * 60)
    print("Test 3: Integración con código existente")
    print("=" * 60)
    
    from src.utils import Filtration_plano_proyectivo
    from src.filtration_graph_builder import construir_grafo_plano_proyectivo
    
    # Construir grafo
    G, VALID_EDGES, POSITIONS, TRIANGLES = construir_grafo_plano_proyectivo()
    
    # Crear instancia de Filtration_plano_proyectivo
    st = Filtration_plano_proyectivo(VALID_EDGES, POSITIONS)
    print(f"✓ Filtration_plano_proyectivo creado correctamente")
    
    # Insertar algunos simplices
    st.insert([1], 0)
    st.insert([2], 0)
    st.insert([1, 2], 1)
    print(f"✓ Simplices insertados: {len(st.simplices)} total")
    
    # Probar distance_matriz
    dist = st.distance_matriz()
    assert dist is not None, "distance_matriz retornó None"
    assert dist.shape == (6, 6), f"Forma incorrecta de matriz: {dist.shape}"
    print(f"✓ distance_matriz funciona: matriz {dist.shape}")
    
    print("✓ Test 3 completado exitosamente\n")
    return True


def run_all_tests():
    """Ejecuta todos los tests."""
    print("\n" + "=" * 60)
    print("EJECUTANDO SUITE DE PRUEBAS")
    print("Aplicación de Filtración Interactiva - Plano Proyectivo")
    print("=" * 60 + "\n")
    
    resultados = []
    
    try:
        resultados.append(("Test 1: Construcción del grafo", test_filtration_graph_builder()))
    except Exception as e:
        print(f"✗ Test 1 fallido: {e}\n")
        resultados.append(("Test 1: Construcción del grafo", False))
    
    try:
        resultados.append(("Test 2: Filtración interactiva", test_interactive_filtration()))
    except Exception as e:
        print(f"✗ Test 2 fallido: {e}\n")
        resultados.append(("Test 2: Filtración interactiva", False))
    
    try:
        resultados.append(("Test 3: Integración", test_integration_with_existing_code()))
    except Exception as e:
        print(f"✗ Test 3 fallido: {e}\n")
        resultados.append(("Test 3: Integración", False))
    
    # Resumen
    print("=" * 60)
    print("RESUMEN DE RESULTADOS")
    print("=" * 60)
    for nombre, resultado in resultados:
        simbolo = "✓" if resultado else "✗"
        estado = "PASÓ" if resultado else "FALLÓ"
        print(f"{simbolo} {nombre}: {estado}")
    
    total = len(resultados)
    exitosos = sum(1 for _, r in resultados if r)
    print(f"\nTotal: {exitosos}/{total} tests exitosos")
    
    if exitosos == total:
        print("\n🎉 ¡Todos los tests pasaron exitosamente!")
        print("\nPuedes ejecutar la aplicación con:")
        print("  python run_projective_plane.py")
    else:
        print("\n⚠ Algunos tests fallaron. Revisa los errores arriba.")
    
    print("=" * 60 + "\n")


if __name__ == '__main__':
    run_all_tests()
