"""
Aplicación interactiva Dash para construir filtraciones del plano proyectivo.

Esta aplicación permite seleccionar simplices interactivamente y visualizar
la homología persistente resultante usando diferentes métodos.
"""

import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import copy

# Importar módulos del proyecto
from src.filtration_graph_builder import (
    construir_grafo_plano_proyectivo,
    extraer_todos_los_simplices,
    nodo_a_id,
    id_a_nodos,
    obtener_caras,
    es_cara_de
)


class ConstructorFiltracionInteractivo:
    """
    Clase principal para manejar el estado y la lógica de la aplicación Dash.
    """
    
    def __init__(self, tamano_filtracion=6):
        """
        Inicializa el constructor de filtración.
        
        Args:
            tamano_filtracion: número de pasos en la filtración (K₀ a K_{n-1})
        """
        self.tamano_filtracion = tamano_filtracion
        
        # Construir el grafo del plano proyectivo
        self.G, self.VALID_EDGES, self.POSITIONS, self.TRIANGLES = construir_grafo_plano_proyectivo()
        
        # Extraer todos los simplices disponibles (en IDs conceptuales)
        self.vertices_ids, self.aristas_ids, self.triangulos_ids = extraer_todos_los_simplices(
            self.G, self.VALID_EDGES, self.TRIANGLES
        )
        
        # Estado inicial: ningún simplex activo
        self.estado_inicial = self.crear_estado_inicial(tamano_filtracion)

    def crear_estado_inicial(self, tamano_filtracion=None):
        """Genera un estado inicial vacío para el tamaño pedido."""
        if tamano_filtracion is None:
            tamano_filtracion = self.tamano_filtracion
        return {
            'tamano_filtracion': tamano_filtracion,
            'paso_actual': 0,
            'simplices': {str(i): [] for i in range(tamano_filtracion)},
            'agregados_en_paso': {str(i): [] for i in range(tamano_filtracion)}
        }
    
    def activar_simplex(self, simplex, en_paso, estado):
        """
        Activa un simplex en el paso especificado y lo propaga hacia adelante.
        También activa recursivamente todas sus caras.
        
        Args:
            simplex: lista de IDs [1, 2] o [1, 2, 3]
            en_paso: índice del paso K_i donde se agrega
            estado: diccionario de estado actual
        
        Returns:
            Estado actualizado
        """
        simplex_ordenado = sorted(simplex)
        en_paso_str = str(en_paso)
        
        # DEBUG
        print(f"\n🔹 Activando {simplex_ordenado} en K_{en_paso}")
        
        # 1. Agregar todas las caras necesarias
        caras = obtener_caras(simplex_ordenado)
        for cara in caras:
            cara_ordenada = sorted(cara)
            if cara_ordenada not in estado['simplices'][en_paso_str]:
                estado['simplices'][en_paso_str].append(cara_ordenada)
                if cara_ordenada not in estado['agregados_en_paso'][en_paso_str]:
                    estado['agregados_en_paso'][en_paso_str].append(cara_ordenada)
                print(f"  ✓ Agregada cara {cara_ordenada} en K_{en_paso}")
        
        # 2. Agregar el simplex mismo
        if simplex_ordenado not in estado['simplices'][en_paso_str]:
            estado['simplices'][en_paso_str].append(simplex_ordenado)
            estado['agregados_en_paso'][en_paso_str].append(simplex_ordenado)
            print(f"  ✓ Agregado simplex {simplex_ordenado} en K_{en_paso}")
        
        # 3. Propagar a pasos futuros
        for t in range(en_paso + 1, estado['tamano_filtracion']):
            t_str = str(t)
            # Propagar el simplex
            if simplex_ordenado not in estado['simplices'][t_str]:
                estado['simplices'][t_str].append(simplex_ordenado)
                print(f"  ↪ Propagado {simplex_ordenado} a K_{t}")
            # Propagar también las caras
            for cara in caras:
                cara_ordenada = sorted(cara)
                if cara_ordenada not in estado['simplices'][t_str]:
                    estado['simplices'][t_str].append(cara_ordenada)
                    print(f"  ↪ Propagada cara {cara_ordenada} a K_{t}")
        
        print(f"✅ Activación completa. K_{en_paso} ahora tiene {len(estado['simplices'][en_paso_str])} simplices")
        return estado
    
    def desactivar_simplex(self, simplex, desde_paso, estado):
        """
        Desactiva un simplex desde el paso especificado en adelante.
        También desactiva recursivamente todas sus cocaras dependientes.
        
        Args:
            simplex: lista de IDs a desactivar
            desde_paso: paso desde el cual desactivar
            estado: diccionario de estado actual
        
        Returns:
            Estado actualizado
        """
        simplex_ordenado = sorted(simplex)
        
        # 1. Primero desactivar todas las cocaras que dependen de este simplex
        for t in range(desde_paso, estado['tamano_filtracion']):
            t_str = str(t)
            # Buscar cocaras en este paso
            cocaras_a_remover = []
            for s in estado['simplices'][t_str]:
                if s != simplex_ordenado and es_cara_de(simplex_ordenado, s):
                    cocaras_a_remover.append(s)
            
            # Remover cocaras recursivamente
            for cocara in cocaras_a_remover:
                estado = self.desactivar_simplex(cocara, t, estado)
        
        # 2. Ahora remover el simplex mismo
        for t in range(desde_paso, estado['tamano_filtracion']):
            t_str = str(t)
            if simplex_ordenado in estado['simplices'][t_str]:
                estado['simplices'][t_str].remove(simplex_ordenado)
            if simplex_ordenado in estado['agregados_en_paso'][t_str]:
                estado['agregados_en_paso'][t_str].remove(simplex_ordenado)
        
        return estado
    
    def esta_activo(self, simplex, en_paso, estado):
        """
        Verifica si un simplex está activo en un paso dado.
        
        Args:
            simplex: lista de IDs
            en_paso: índice del paso
            estado: diccionario de estado
        
        Returns:
            True si el simplex está activo en ese paso
        """
        simplex_ordenado = sorted(simplex)
        return simplex_ordenado in estado['simplices'][str(en_paso)]
    
    def crear_figura_filtracion(self, estado):
        """
        Crea la figura completa de la filtración con todos los pasos lado a lado.
        
        Args:
            estado: diccionario de estado actual
        
        Returns:
            Figura Plotly
        """
        num_pasos = estado['tamano_filtracion']
        
        # DEBUG: Mostrar estado completo
        print(f"\n🎨 Creando figura de filtración:")
        for i in range(num_pasos):
            count = len(estado['simplices'][str(i)])
            print(f"   K_{i}: {count} simplices")
        
        # Crear subplots
        fig = make_subplots(
            rows=1, 
            cols=num_pasos,
            subplot_titles=[f"K{i}" for i in range(num_pasos)],
            horizontal_spacing=0.05
        )
        
        # Para cada paso de la filtración
        for paso in range(num_pasos):
            col = paso + 1
            
            # Capa 1: Dibujar todos los simplices en gris (inactivos)
            self._agregar_capa_inactiva(fig, paso, col)
            
            # Capa 2: Dibujar simplices activos en azul (sobrepuestos)
            self._agregar_capa_activa(fig, paso, col, estado)
        
        # Configurar layout
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
        
        fig.update_layout(
            height=400,
            showlegend=False,
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    def _agregar_capa_inactiva(self, fig, paso, col):
        """
        Agrega la capa de simplices inactivos (gris, baja opacidad).
        """
        # Dibujar todos los triángulos en gris
        for triangulo_ids in self.triangulos_ids:
            self._dibujar_triangulo(
                fig,
                triangulo_ids,
                col,
                fillcolor='rgba(189, 195, 199, 0.18)',
                line_color='rgba(127, 140, 141, 0.35)'
            )
        
        # Dibujar todas las aristas en gris
        for arista_ids in self.aristas_ids:
            self._dibujar_arista(fig, arista_ids, col, color='lightgray', width=1, opacity=0.3)
        
        # Dibujar todos los vértices en gris
        for vertice_id in self.vertices_ids:
            self._dibujar_vertice(
                fig,
                vertice_id,
                col,
                color='lightgray',
                size=8,
                opacity=0.35,
                mostrar_etiqueta=True
            )
    
    def _agregar_capa_activa(self, fig, paso, col, estado):
        """
        Agrega la capa de simplices activos (azul, opacidad completa).
        """
        simplices_activos = estado['simplices'][str(paso)]
        
        # DEBUG
        print(f"  📊 K_{paso} (col={col}): {len(simplices_activos)} simplices activos: {simplices_activos}")
        
        # Separar por dimensión
        triangulos_activos = [s for s in simplices_activos if len(s) == 3]
        aristas_activas = [s for s in simplices_activos if len(s) == 2]
        vertices_activos = [s for s in simplices_activos if len(s) == 1]
        
        # Dibujar triángulos activos
        for triangulo in triangulos_activos:
            self._dibujar_triangulo(
                fig,
                triangulo,
                col,
                fillcolor='rgba(52, 152, 219, 0.38)',
                line_color='rgba(41, 128, 185, 0.75)',
                line_width=1
            )
        
        # Dibujar aristas activas
        for arista in aristas_activas:
            self._dibujar_arista(fig, arista, col, color='blue', width=3, opacity=1.0)
        
        # Dibujar vértices activos
        for vertice in vertices_activos:
            self._dibujar_vertice(
                fig,
                vertice,
                col,
                color='blue',
                size=12,
                opacity=1.0,
                mostrar_etiqueta=False
            )

    def _dibujar_vertice(self, fig, vertice_id, col, color='blue', size=10, opacity=1.0, mostrar_etiqueta=True):
        """Dibuja un vértice en el subplot especificado."""
        nodos_fisicos = id_a_nodos(vertice_id[0])

        for nodo in nodos_fisicos:
            pos = self.POSITIONS[nodo]
            modo = 'markers+text' if mostrar_etiqueta else 'markers'
            trace_kwargs = dict(
                x=[pos[0]],
                y=[pos[1]],
                mode=modo,
                marker=dict(size=size, color=color, opacity=opacity),
                hoverinfo='text',
                hovertext=f"ID: {vertice_id[0]} (nodo {nodo})",
                showlegend=False
            )
            if mostrar_etiqueta:
                trace_kwargs['text'] = [str(nodo_a_id(nodo))]
                trace_kwargs['textposition'] = 'top center'
                trace_kwargs['textfont'] = dict(color='#2c3e50')

            fig.add_trace(
                go.Scatter(**trace_kwargs),
                row=1, col=col
            )

    def _dibujar_arista(self, fig, arista_ids, col, color='blue', width=2, opacity=1.0):
        """Dibuja una arista en el subplot especificado."""
        id1, id2 = arista_ids
        nodos1 = id_a_nodos(id1)
        nodos2 = id_a_nodos(id2)

        for nodo1 in nodos1:
            for nodo2 in nodos2:
                if (nodo1, nodo2) in self.VALID_EDGES or (nodo2, nodo1) in self.VALID_EDGES:
                    pos1 = self.POSITIONS[nodo1]
                    pos2 = self.POSITIONS[nodo2]

                    fig.add_trace(
                        go.Scatter(
                            x=[pos1[0], pos2[0]],
                            y=[pos1[1], pos2[1]],
                            mode='lines',
                            line=dict(color=color, width=width),
                            opacity=opacity,
                            hoverinfo='text',
                            hovertext=f"Arista: {arista_ids}",
                            showlegend=False
                        ),
                        row=1, col=col
                    )

    def _dibujar_triangulo(self, fig, triangulo_ids, col, fillcolor='rgba(52, 152, 219, 0.38)', line_color=None, line_width=0):
        """Dibuja un triángulo en el subplot especificado."""
        id1, id2, id3 = triangulo_ids
        nodos1 = id_a_nodos(id1)
        nodos2 = id_a_nodos(id2)
        nodos3 = id_a_nodos(id3)

        actual_line_color = line_color or fillcolor

        for n1 in nodos1:
            for n2 in nodos2:
                for n3 in nodos3:
                    arista_12 = (n1, n2) in self.VALID_EDGES or (n2, n1) in self.VALID_EDGES
                    arista_23 = (n2, n3) in self.VALID_EDGES or (n3, n2) in self.VALID_EDGES
                    arista_31 = (n3, n1) in self.VALID_EDGES or (n1, n3) in self.VALID_EDGES

                    if arista_12 and arista_23 and arista_31:
                        pos1 = self.POSITIONS[n1]
                        pos2 = self.POSITIONS[n2]
                        pos3 = self.POSITIONS[n3]

                        fig.add_trace(
                            go.Scatter(
                                x=[pos1[0], pos2[0], pos3[0], pos1[0]],
                                y=[pos1[1], pos2[1], pos3[1], pos1[1]],
                                mode='lines',
                                fill='toself',
                                fillcolor=fillcolor,
                                line=dict(color=actual_line_color, width=line_width),
                                hoverinfo='text',
                                hovertext=f"Triángulo: {triangulo_ids}",
                                showlegend=False
                            ),
                            row=1, col=col
                        )


def _generar_botones_iniciales(constructor):
    """
    Genera los botones iniciales para K_0 (paso 0).
    
    Args:
        constructor: instancia de ConstructorFiltracionInteractivo
    
    Returns:
        Lista de elementos HTML con los botones
    """
    botones = []
    
    def estilo_boton_inactivo():
        return {
            'margin': '5px',
            'backgroundColor': '#95a5a6',  # Gris (inactivo)
            'color': 'white',
            'border': 'none',
            'padding': '8px 15px',
            'borderRadius': '5px',
            'cursor': 'pointer'
        }

    # Vértices
    botones.append(html.H4("Vértices:"))
    div_vertices = []
    for i, vertice in enumerate(constructor.vertices_ids):
        div_vertices.append(html.Button(
            f"{vertice[0]}",
            id={'type': 'simplex-btn', 'dim': 0, 'index': i},
            n_clicks=0,
            style=estilo_boton_inactivo()
        ))
    botones.append(html.Div(div_vertices, style={'marginBottom': '15px'}))

    # Aristas
    botones.append(html.H4("Aristas:"))
    div_aristas = []
    for i, arista in enumerate(constructor.aristas_ids):
        div_aristas.append(html.Button(
            f"{tuple(arista)}",
            id={'type': 'simplex-btn', 'dim': 1, 'index': i},
            n_clicks=0,
            style=estilo_boton_inactivo()
        ))
    botones.append(html.Div(div_aristas, style={'marginBottom': '15px'}))

    # Triángulos
    botones.append(html.H4("Triángulos:"))
    div_triangulos = []
    for i, triangulo in enumerate(constructor.triangulos_ids):
        div_triangulos.append(html.Button(
            f"{tuple(triangulo)}",
            id={'type': 'simplex-btn', 'dim': 2, 'index': i},
            n_clicks=0,
            style=estilo_boton_inactivo()
        ))
    botones.append(html.Div(div_triangulos, style={'marginBottom': '15px'}))

    return botones


def crear_app_dash(tamano_filtracion=6):
    """
    Crea y configura la aplicación Dash.
    
    Args:
        tamano_filtracion: tamaño de la filtración
    
    Returns:
        Aplicación Dash configurada
    """
    constructor = ConstructorFiltracionInteractivo(tamano_filtracion)
    app = dash.Dash(__name__)
    
    # Layout de la aplicación
    app.layout = html.Div([
        html.H1("Filtración Interactiva - Plano Proyectivo", 
                style={'textAlign': 'center', 'color': '#2c3e50'}),
        
        # Almacenamiento del estado
        dcc.Store(id='estado-filtracion', data=constructor.estado_inicial),
        
        # Gráfico de la filtración
        html.Div([
            dcc.Graph(
                id='grafico-filtracion',
                figure=constructor.crear_figura_filtracion(constructor.estado_inicial),
                config={
                    'displayModeBar': False,
                    'scrollZoom': False,
                    'doubleClick': False,
                    'staticPlot': True  # Desactivar todas las interacciones del gráfico
                }
            )
        ]),
        
        # Controles
        html.Div([
            html.H3("Controles"),
            html.Div([
                html.Label("Paso actual: "),
                html.Span(id='paso-actual-texto', children="K0"),
                html.Button("◀ Anterior", id='btn-anterior', n_clicks=0, 
                           style={'marginLeft': '10px'}),
                html.Button("Siguiente ▶", id='btn-siguiente', n_clicks=0,
                           style={'marginLeft': '10px'}),
            ], style={'marginBottom': '10px'}),
            html.Div([
                html.Label("Tamaño de filtración:"),
                dcc.Input(
                    id='input-filtration-size',
                    type='number',
                    min=1,
                    max=12,
                    step=1,
                    value=tamano_filtracion,
                    style={'marginLeft': '10px', 'width': '80px'}
                ),
            ], style={'marginBottom': '20px'}),
            html.Div([
                dcc.Checklist(
                    id='toggle-estado',
                    options=[{'label': 'Mostrar estado actual', 'value': 'mostrar'}],
                    value=['mostrar'],
                    inputStyle={'marginRight': '6px'},
                    labelStyle={'display': 'inline-flex', 'alignItems': 'center', 'gap': '6px'}
                )
            ], style={'marginBottom': '20px'}),
            
            html.Div([
                html.Label("Simplices disponibles (click para activar/desactivar):"),
                html.Div(id='lista-simplices', children=_generar_botones_iniciales(constructor))
            ], style={'marginBottom': '20px'}),
        ], style={'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '10px',
                  'margin': '20px'}),
        
        # Estado actual
        html.Div([
            html.H3("Estado Actual"),
            html.Div(id='resumen-estado', children=[])
        ], id='contenedor-estado', hidden=False,
           style={'padding': '20px', 'backgroundColor': '#e8f8f5', 'borderRadius': '10px',
                  'margin': '20px'}),
        
        # Acciones
        html.Div([
            html.Button("📋 Exportar Código SIMPLICES", id='btn-exportar', n_clicks=0,
                       style={'marginRight': '10px'}),
            html.Button("📊 Calcular Homología Persistente", id='btn-calcular', n_clicks=0),
            html.Div(id='output-exportar', style={'marginTop': '10px'})
        ], style={'padding': '20px', 'textAlign': 'center'}),
        
        # Referencia al constructor (guardado en atributo)
        html.Div(id='constructor-ref', style={'display': 'none'})
    ])
    
    # Almacenar constructor en el servidor
    app.constructor = constructor
    
    # ===== CALLBACKS =====
    
    @app.callback(
        [Output('estado-filtracion', 'data'),
         Output('grafico-filtracion', 'figure'),
         Output('paso-actual-texto', 'children')],
        [Input('btn-anterior', 'n_clicks'),
         Input('btn-siguiente', 'n_clicks')],
        [State('estado-filtracion', 'data')]
    )
    def navegar_pasos(n_ant, n_sig, estado):
        """Navega entre los pasos de la filtración."""
        ctx = callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        
        # Hacer una copia profunda del estado para evitar mutaciones
        estado = copy.deepcopy(estado)
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # DEBUG
        print(f"\n🔄 Navegando desde K_{estado['paso_actual']}")
        
        if button_id == 'btn-anterior' and estado['paso_actual'] > 0:
            estado['paso_actual'] -= 1
        elif button_id == 'btn-siguiente' and estado['paso_actual'] < estado['tamano_filtracion'] - 1:
            estado['paso_actual'] += 1
        else:
            raise dash.exceptions.PreventUpdate
        
        print(f"   → Ahora en K_{estado['paso_actual']}")
        
        fig = constructor.crear_figura_filtracion(estado)
        texto = f"K{estado['paso_actual']}"
        
        return estado, fig, texto
    
    @app.callback(
        Output('lista-simplices', 'children'),
        Input('estado-filtracion', 'data')
    )
    def actualizar_lista_simplices(estado):
        """Actualiza la lista de simplices disponibles en el paso actual."""
        paso_actual = estado['paso_actual']
        paso_actual_str = str(paso_actual)
        
        # Crear botones para cada tipo de simplex
        botones = []
        
        # Vértices
        botones.append(html.H4("Vértices:"))
        div_vertices = []
        for i, vertice in enumerate(constructor.vertices_ids):
            activo = vertice in estado['simplices'][paso_actual_str]
            estilo = {
                'margin': '5px',
                'backgroundColor': '#3498db' if activo else '#95a5a6',
                'color': 'white',
                'border': 'none',
                'padding': '8px 15px',
                'borderRadius': '5px',
                'cursor': 'pointer'
            }
            div_vertices.append(html.Button(
                f"{vertice[0]} ✓" if activo else f"{vertice[0]}",
                id={'type': 'simplex-btn', 'dim': 0, 'index': i},
                n_clicks=0,
                style=estilo
            ))
        botones.append(html.Div(div_vertices, style={'marginBottom': '15px'}))
        
        # Aristas
        botones.append(html.H4("Aristas:"))
        div_aristas = []
        for i, arista in enumerate(constructor.aristas_ids):
            activo = arista in estado['simplices'][paso_actual_str]
            estilo = {
                'margin': '5px',
                'backgroundColor': '#3498db' if activo else '#95a5a6',
                'color': 'white',
                'border': 'none',
                'padding': '8px 15px',
                'borderRadius': '5px',
                'cursor': 'pointer'
            }
            div_aristas.append(html.Button(
                f"{tuple(arista)} ✓" if activo else f"{tuple(arista)}",
                id={'type': 'simplex-btn', 'dim': 1, 'index': i},
                n_clicks=0,
                style=estilo
            ))
        botones.append(html.Div(div_aristas, style={'marginBottom': '15px'}))
        
        # Triángulos
        botones.append(html.H4("Triángulos:"))
        div_triangulos = []
        for i, triangulo in enumerate(constructor.triangulos_ids):
            activo = triangulo in estado['simplices'][paso_actual_str]
            estilo = {
                'margin': '5px',
                'backgroundColor': '#3498db' if activo else '#95a5a6',
                'color': 'white',
                'border': 'none',
                'padding': '8px 15px',
                'borderRadius': '5px',
                'cursor': 'pointer'
            }
            div_triangulos.append(html.Button(
                f"{tuple(triangulo)} ✓" if activo else f"{tuple(triangulo)}",
                id={'type': 'simplex-btn', 'dim': 2, 'index': i},
                n_clicks=0,
                style=estilo
            ))
        botones.append(html.Div(div_triangulos, style={'marginBottom': '15px'}))
        
        return botones
    
    @app.callback(
        [Output('estado-filtracion', 'data', allow_duplicate=True),
         Output('grafico-filtracion', 'figure', allow_duplicate=True)],
        Input({'type': 'simplex-btn', 'dim': dash.dependencies.ALL, 'index': dash.dependencies.ALL}, 'n_clicks'),
        State('estado-filtracion', 'data'),
        prevent_initial_call=True
    )
    def toggle_simplex(n_clicks_list, estado):
        """Activa o desactiva un simplex cuando se hace click."""
        ctx = callback_context
        
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        
        # Evitar disparos falsos cuando Dash recrea los botones (n_clicks vuelve a 0)
        trigger_value = ctx.triggered[0].get('value', None)
        if not trigger_value:
            raise dash.exceptions.PreventUpdate

        # Hacer una copia profunda del estado para evitar mutaciones
        estado = copy.deepcopy(estado)
        
        # Obtener qué botón fue clickeado
        trigger_id = ctx.triggered_id
        if trigger_id is None:
            raise dash.exceptions.PreventUpdate
        
        dim = trigger_id['dim']
        index = trigger_id['index']
        
        # Obtener el simplex correspondiente
        if dim == 0:
            simplex = constructor.vertices_ids[index]
        elif dim == 1:
            simplex = constructor.aristas_ids[index]
        elif dim == 2:
            simplex = constructor.triangulos_ids[index]
        else:
            raise dash.exceptions.PreventUpdate
        
        paso_actual = estado['paso_actual']
        paso_actual_str = str(paso_actual)
        
        # Toggle: si está activo, desactivar; si no, activar
        if simplex in estado['simplices'][paso_actual_str]:
            # Desactivar
            estado = constructor.desactivar_simplex(simplex, paso_actual, estado)
        else:
            # Activar
            estado = constructor.activar_simplex(simplex, paso_actual, estado)
        
        # Actualizar figura
        fig = constructor.crear_figura_filtracion(estado)
        
        return estado, fig
    
    @app.callback(
        Output('resumen-estado', 'children'),
        Input('estado-filtracion', 'data')
    )
    def actualizar_resumen(estado):
        """Actualiza el resumen del estado actual."""
        paso_actual = estado['paso_actual']
        paso_actual_str = str(paso_actual)
        
        # Contar simplices
        nuevos = estado['agregados_en_paso'][paso_actual_str]
        todos = estado['simplices'][paso_actual_str]
        heredados = [s for s in todos if s not in nuevos]
        
        # Separar por dimensión
        nuevos_vertices = [s for s in nuevos if len(s) == 1]
        nuevos_aristas = [s for s in nuevos if len(s) == 2]
        nuevos_triangulos = [s for s in nuevos if len(s) == 3]
        
        heredados_vertices = [s for s in heredados if len(s) == 1]
        heredados_aristas = [s for s in heredados if len(s) == 2]
        heredados_triangulos = [s for s in heredados if len(s) == 3]
        
        return html.Div([
            html.H4(f"K{paso_actual} - Nuevos en este paso ({len(nuevos)} simplices):"),
            html.P(f"Vértices: {nuevos_vertices}" if nuevos_vertices else "Vértices: ninguno"),
            html.P(f"Aristas: {nuevos_aristas}" if nuevos_aristas else "Aristas: ninguno"),
            html.P(f"Triángulos: {nuevos_triangulos}" if nuevos_triangulos else "Triángulos: ninguno"),
            html.Hr(),
            html.H4(f"K{paso_actual} - Heredados ({len(heredados)} simplices):"),
            html.P(f"Vértices: {len(heredados_vertices)}"),
            html.P(f"Aristas: {len(heredados_aristas)}"),
            html.P(f"Triángulos: {len(heredados_triangulos)}")
        ])

    @app.callback(
        Output('contenedor-estado', 'hidden'),
        Input('toggle-estado', 'value')
    )
    def mostrar_u_ocultar_estado(valores):
        """Permite ocultar el panel de estado si el usuario lo desea."""
        if valores and 'mostrar' in valores:
            return False
        return True
    
    @app.callback(
        Output('output-exportar', 'children'),
        Input('btn-exportar', 'n_clicks'),
        State('estado-filtracion', 'data')
    )
    def exportar_simplices(n_clicks, estado):
        """Exporta el código SIMPLICES."""
        if n_clicks == 0:
            return ""
        
        codigo = f"FILTRATION_SIZE = {estado['tamano_filtracion']}\n"
        codigo += "SIMPLICES = {\n"
        
        for i in range(estado['tamano_filtracion']):
            nuevos = estado['agregados_en_paso'][str(i)]
            codigo += f"    {i}: {nuevos},\n"
        
        codigo += "}\n"
        
        return html.Pre(codigo, style={
            'backgroundColor': '#2c3e50',
            'color': '#ecf0f1',
            'padding': '15px',
            'borderRadius': '5px',
            'overflow': 'auto',
            'marginTop': '10px'
        })
    
    @app.callback(
        [Output('estado-filtracion', 'data', allow_duplicate=True),
         Output('grafico-filtracion', 'figure', allow_duplicate=True),
         Output('paso-actual-texto', 'children', allow_duplicate=True)],
        Input('input-filtration-size', 'value'),
        prevent_initial_call=True
    )
    def actualizar_tamano_filtracion(valor):
        """Permite al usuario redefinir el tamaño de la filtración."""
        if valor is None or valor < 1:
            raise dash.exceptions.PreventUpdate

        nuevo_tamano = int(valor)
        constructor.tamano_filtracion = nuevo_tamano
        nuevo_estado = constructor.crear_estado_inicial(nuevo_tamano)
        constructor.estado_inicial = copy.deepcopy(nuevo_estado)

        fig = constructor.crear_figura_filtracion(nuevo_estado)
        return nuevo_estado, fig, "K0"

    return app


if __name__ == '__main__':
    app = crear_app_dash(tamano_filtracion=6)
    app.run(debug=True, port=8050)
