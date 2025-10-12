# DataTopology-2025

## App Interactiva de Filtración - Plano Proyectivo

### 📋 Descripción

Esta aplicación interactiva permite construir filtraciones del plano proyectivo de manera visual e interactiva, y calcular la homología persistente usando diferentes métodos.

### 🚀 Inicio Rápido

```bash
python run_projective_plane.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8050`

### 🎯 Características

#### 1. Visualización Interactiva
- Muestra todos los pasos de la filtración (K₀ a K₅) lado a lado
- Simplices inactivos se muestran en gris con baja opacidad
- Simplices activos se muestran en azul con opacidad completa

#### 2. Construcción de Filtración
- Click en simplices para activar/desactivar
- Propagación automática a pasos futuros
- Manejo automático de caras y cocaras

#### 3. Cálculo de Homología Persistente
- Algoritmo personalizado sobre ℚ
- GUDHI con matriz de distancias sobre 𝔽₂
- Visualización de diagramas de persistencia y códigos de barras

#### 4. Exportación
- Exportar código SIMPLICES para usar en notebooks
- Guardar/cargar configuraciones (próximamente)