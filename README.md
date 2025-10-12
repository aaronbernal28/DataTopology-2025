# DataTopology-2025

## App Interactiva de Filtración - Plano Proyectivo

### 📋 Descripción

Aplicación Dash que facilita la construcción paso a paso de filtraciones sobre el plano proyectivo, mostrando en tiempo real la evolución de los simplices y los resultados de homología persistente calculados con distintos coeficientes.

### 🚀 Inicio Rápido

```bash
python run_projective_plane.py
```

1. Crea y activa un entorno virtual (opcional pero recomendado).
2. Instala las dependencias con `pip install -r requirements.txt`.
3. Ejecuta el script anterior desde la raíz del proyecto.

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8050`.

### 🎯 Características

#### 1. Visualización Interactiva
- Panel principal con todos los pasos de la filtración (K₀ a Kₙ).

#### 2. Construcción de Filtración
- Con un clic sobre vértices, aristas o triángulos se activan o desactivan en el paso seleccionado.
- Las activaciones se propagan de manera automática a pasos posteriores y garantizan que las caras requeridas estén presentes.
- Controles dedicados permiten navegar entre pasos, ajustar el tamaño máximo de la filtración y cargar un ejemplo preconfigurado.

#### 3. Cálculo de Homología Persistente
- Algoritmo personalizado sobre ℚ
- GUDHI con matriz de distancias sobre 𝔽₂
- Visualización de diagramas de persistencia y códigos de barras

#### 4. Exportación
- Exportar código SIMPLICES para usar en notebooks
- Guardar/cargar configuraciones (próximamente)
- Clickear directamente en el grafo para editarlos (próximamente)