"""
Script de lanzamiento para la aplicación interactiva de filtración del plano proyectivo.

Ejecuta este script para abrir la aplicación Dash en el navegador.
"""

import sys
import os
import webbrowser
import threading

# Asegurar que el directorio raíz está en el path
directorio_raiz = os.path.dirname(os.path.abspath(__file__))
if directorio_raiz not in sys.path:
    sys.path.insert(0, directorio_raiz)

from src.interactive_filtration_dash import crear_app_dash


def main():
    """
    Función principal para iniciar la aplicación.
    """
    print("=" * 60)
    print("Iniciando Aplicación de Filtración Interactiva")
    print("Plano Proyectivo - Homología Persistente")
    print("=" * 60)
    print()
    print("La aplicación se abrirá en tu navegador en:")
    print("  http://localhost:8050")
    print()
    print("Para detener la aplicación, presiona Ctrl+C")
    print("=" * 60)
    print()
    
    # Crear y lanzar la aplicación
    app = crear_app_dash(tamano_filtracion=6)

    def _abrir_navegador():
        try:
            webbrowser.open_new("http://localhost:8050")
        except Exception:
            pass

    threading.Timer(1.0, _abrir_navegador).start()
    app.run(debug=True, port=8050)


if __name__ == '__main__':
    main()
