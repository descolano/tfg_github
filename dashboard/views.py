from django.shortcuts import render
from django.db import connection
from django.http import JsonResponse
from .models import Dm2Comorbilidades, Dm2Complicaciones, Dm2Departamentos, Dm2EdadSexo, PredecirFarmacia, Dm2CostesTotales, PredecirCosteTotal
from datetime import date
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import plotly.graph_objects as go
import plotly.offline as opy
import pickle
import base64

def dashboard_home(request):
    return render(request, 'dashboard_home.html')

def prevalencia_edad_sexo(request):
    try:
        # Opción 1: Intentar con modelo Django primero
        datos = list(Dm2EdadSexo.objects.all().values(
            'edad', 
            'pacientes_hombre', 
            'prevalencia_hombre', 
            'pacientes_mujer',                                        
            'prevalencia_mujer'
        ))
        
        # Si no hay datos con el modelo, usar consulta SQL directa
        if not datos:
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT edad, pacientes_hombre, prevalencia_hombre, 
                           pacientes_mujer, prevalencia_mujer 
                    FROM dm2_por_edad_y_sexo 
                    ORDER BY edad
                """)
                
                columns = ['edad', 'pacientes_hombre', 'prevalencia_hombre', 'pacientes_mujer', 'prevalencia_mujer']
                datos = []
                for row in cursor.fetchall():
                    datos.append(dict(zip(columns, row)))
        
        # Crear gráfico con Plotly si hay datos
        grafico_html = None
        if datos:
            # Extraer datos para el gráfico
            edades = [d['edad'] for d in datos]
            
            # Convertir prevalencias de texto a números
            hombres_prev = []
            mujeres_prev = []
            
            for d in datos:
                # Procesar prevalencia hombres
                prev_h = str(d['prevalencia_hombre']).replace('%', '').strip()
                try:
                    hombres_prev.append(float(prev_h))
                except:
                    hombres_prev.append(0)
                
                # Procesar prevalencia mujeres
                prev_m = str(d['prevalencia_mujer']).replace('%', '').strip()
                try:
                    mujeres_prev.append(float(prev_m))
                except:
                    mujeres_prev.append(0)
            
            # Crear gráfico de barras agrupadas
            fig = go.Figure()
            
            # Añadir barras de hombres
            fig.add_trace(go.Bar(
                name='Hombre',
                x=edades,
                y=hombres_prev,
                marker_color='#4A90E2',  # Azul
                text=[f'{val:.2f}%' for val in hombres_prev],
                textposition='outside',
                textfont=dict(size=10, color='#4A90E2')
            ))
            
            # Añadir barras de mujeres
            fig.add_trace(go.Bar(
                name='Mujer',
                x=edades,
                y=mujeres_prev,
                marker_color='#e91e63',  # Rosa/rojo consistente con tabla
                text=[f'{val:.2f}%' for val in mujeres_prev],
                textposition='outside',
                textfont=dict(size=10, color='#e91e63')
            ))
            
            # Configuración layout
            fig.update_layout(
                title={
                    'text': 'Prevalencia de DM2 por Edad y Sexo',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16}
                },
                xaxis_title='Edad (rangos)',
                yaxis_title='Prevalencia (%)',
                barmode='group',
                height=500,
                width=900,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.25,
                    xanchor="center",
                    x=0.5,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.1)",
                    borderwidth=1
                ),
                # Configurar ejes
                xaxis=dict(
                    title_font=dict(size=12),
                    tickfont=dict(size=11),
                    title_standoff=20
                ),
                yaxis=dict(
                    title_font=dict(size=12),
                    tickfont=dict(size=11),
                    range=[0, max(max(hombres_prev), max(mujeres_prev)) * 1.15]
                ),
                # Colores de fondo
                paper_bgcolor='white',
                plot_bgcolor='white',
                # Líneas de cuadrícula
                yaxis_showgrid=True,
                yaxis_gridcolor='lightgray',
                yaxis_gridwidth=0.5,
                # Márgenes ajustados
                margin=dict(
                    l=60,
                    r=50, 
                    t=80,
                    b=100
                )
            )
            
            # Convertir a HTML
            grafico_html = opy.plot(
                fig, 
                output_type='div', 
                include_plotlyjs=True
            )
        
        return render(request, 'prevalencia_edad_sexo.html', {
            'datos': datos,
            'grafico': grafico_html
        })
        
    except Exception as e:
        print(f"Error en prevalencia_edad_sexo: {e}")
        return render(request, 'prevalencia_edad_sexo.html', {'error': str(e)})

def comorbilidades(request):
    try:
        # Función para transformar nombres de comorbilidades
        def transformar_nombre_comorbilidad(nombre):
            mapeo_comorbilidades = {
                'hta': 'Hipertensión arterial',
                'fa': 'Fibrilación auricular', 
                'ic': 'Insuficiencia cardíaca',
                'dislipemias': 'Dislipemias',
                'obesidad': 'Obesidad',
                'ansiedad': 'Ansiedad',
                'insomnio': 'Insomnio',
                'depresion': 'Depresión',
                'demencia': 'Demencia',
                'alzheimer': 'Alzheimer',
                'artrosis': 'Artrosis',
                'epoc': 'EPOC',
                'asma': 'Asma',
                'osteoporosis': 'Osteoporosis'
            }
            return mapeo_comorbilidades.get(nombre, nombre.replace('_', ' ').title())
        
        # Función para transformar nombres de complicaciones
        def transformar_nombre_complicacion(nombre):
            mapeo_complicaciones = {
                'ic': 'Insuficiencia cardíaca',
                'erc': 'Enfermedad renal crónica',
                'nefropatia_diabetica': 'Nefropatía diabética',
                'retinopatia_diabetica': 'Retinopatía diabética',
                'neuropatia_diabetica': 'Neuropatía diabética', 
                'cardiopatia_isquemica': 'Cardiopatía isquémica',
                'enf_arterial_periferica': 'Enf. arterial periférica',
                'enf_cerebrovascular': 'Enf. cerebrovascular',
                'insuficiencia_venosa': 'Insuficiencia venosa'
            }
            return mapeo_complicaciones.get(nombre, nombre.replace('_', ' ').title())
        
        # Obtener datos de comorbilidades y transformar nombres
        datos_comorbilidades_raw = Dm2Comorbilidades.objects.all().order_by('-prevalencia')
        datos_comorbilidades = []
        for item in datos_comorbilidades_raw:
            # Crear objeto con nombre transformado
            class ComorbillidadTransformada:
                def __init__(self, original):
                    self.comorbilidad = transformar_nombre_comorbilidad(original.comorbilidad)
                    self.comorbilidad_original = original.comorbilidad  # Guardar el original para los gráficos
                    self.pacientes = original.pacientes
                    self.prevalencia = original.prevalencia
            
            datos_comorbilidades.append(ComorbillidadTransformada(item))
        
        # Obtener datos de complicaciones y transformar nombres
        try:
            datos_complicaciones_raw = Dm2Complicaciones.objects.all().order_by('-prevalencia')
            datos_complicaciones = []
            for item in datos_complicaciones_raw:
                # Crear objeto con nombre transformado
                class ComplicacionTransformada:
                    def __init__(self, original):
                        self.complicacion = transformar_nombre_complicacion(original.complicacion)
                        self.complicacion_original = original.complicacion  # Guardar el original para los gráficos
                        self.pacientes = original.pacientes
                        self.prevalencia = original.prevalencia
                
                datos_complicaciones.append(ComplicacionTransformada(item))
        except:
            # Si el modelo no existe, usar consulta SQL directa
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT complicacion, pacientes, prevalencia 
                    FROM dm2_complicaciones 
                    ORDER BY prevalencia DESC
                """)
                
                columns = ['complicacion', 'pacientes', 'prevalencia']
                datos_complicaciones = []
                for row in cursor.fetchall():
                    # Crear objeto similar al modelo con nombre transformado
                    class ComplicacionObj:
                        def __init__(self, complicacion, pacientes, prevalencia):
                            self.complicacion = transformar_nombre_complicacion(complicacion)
                            self.complicacion_original = complicacion
                            self.pacientes = pacientes
                            self.prevalencia = prevalencia
                    
                    datos_complicaciones.append(ComplicacionObj(*row))
        
        # Crear gráfico de comorbilidades
        grafico_comorbilidades_html = None
        if datos_comorbilidades:
            comorbilidades_nombres = [d.comorbilidad for d in datos_comorbilidades]
            comorbilidades_prevalencias = [float(d.prevalencia) if d.prevalencia else 0 for d in datos_comorbilidades]
            
            fig_comorbilidades = go.Figure(go.Bar(
                x=comorbilidades_nombres,
                y=comorbilidades_prevalencias,
                marker_color='#22C55E',  # Verde para comorbilidades
                text=[f'{val:.1f}%' for val in comorbilidades_prevalencias],
                textposition='outside',
                textfont=dict(size=10, color='#16A34A'),
                hovertemplate="<b>%{x}</b><br>" +
                            "Prevalencia: %{y:.2f}%<br>" +
                            "<extra></extra>"
            ))
            
            fig_comorbilidades.update_layout(
                title={
                    'text': 'Prevalencia de Comorbilidades en DM2',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16, 'color': '#2c3e50'}
                },
                xaxis_title='Comorbilidades',
                yaxis_title='Prevalencia (%)',
                height=500,
                width=1000,
                showlegend=False,
                xaxis=dict(
                    title_font=dict(size=12),
                    tickfont=dict(size=12),
                    tickangle=45,
                    automargin=True
                ),
                yaxis=dict(
                    title_font=dict(size=12),
                    tickfont=dict(size=11),
                    range=[0, max(comorbilidades_prevalencias) * 1.15] if comorbilidades_prevalencias else [0, 100]
                ),
                paper_bgcolor='white',
                plot_bgcolor='white',
                yaxis_showgrid=True,
                yaxis_gridcolor='lightgray',
                yaxis_gridwidth=0.5,
                margin=dict(l=60, r=50, t=80, b=120)
            )
            
            grafico_comorbilidades_html = opy.plot(
                fig_comorbilidades, 
                output_type='div', 
                include_plotlyjs=True
            )
        
        # Crear gráfico de complicaciones
        grafico_complicaciones_html = None
        if datos_complicaciones:
            complicaciones_nombres = [d.complicacion for d in datos_complicaciones]
            complicaciones_prevalencias = [float(d.prevalencia) if d.prevalencia else 0 for d in datos_complicaciones]
            
            fig_complicaciones = go.Figure(go.Bar(
                x=complicaciones_nombres,
                y=complicaciones_prevalencias,
                marker_color='#EAB308',
                text=[f'{val:.1f}%' for val in complicaciones_prevalencias],
                textposition='outside',
                textfont=dict(size=10, color='#CA8A04'),
                hovertemplate="<b>%{x}</b><br>" +
                            "Prevalencia: %{y:.2f}%<br>" +
                            "<extra></extra>"
            ))
            
            fig_complicaciones.update_layout(
                title={
                    'text': 'Prevalencia de Complicaciones en DM2',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16, 'color': '#2c3e50'}
                },
                xaxis_title='Complicaciones',
                yaxis_title='Prevalencia (%)',
                height=500,
                width=1000,
                showlegend=False,
                xaxis=dict(
                    title_font=dict(size=12),
                    tickfont=dict(size=12),
                    tickangle=45,
                    automargin=True
                ),
                yaxis=dict(
                    title_font=dict(size=12),
                    tickfont=dict(size=11),
                    range=[0, max(complicaciones_prevalencias) * 1.15] if complicaciones_prevalencias else [0, 100]
                ),
                paper_bgcolor='white',
                plot_bgcolor='white',
                yaxis_showgrid=True,
                yaxis_gridcolor='lightgray',
                yaxis_gridwidth=0.5,
                margin=dict(l=60, r=50, t=80, b=120)
            )
            
            grafico_complicaciones_html = opy.plot(
                fig_complicaciones, 
                output_type='div', 
                include_plotlyjs=True
            )
        
        return render(request, 'comorbilidades.html', {
            'datos_comorbilidades': datos_comorbilidades,
            'datos_complicaciones': datos_complicaciones,
            'grafico_comorbilidades': grafico_comorbilidades_html,
            'grafico_complicaciones': grafico_complicaciones_html,
            'titulo': 'Comorbilidades y Complicaciones DM2'
        })
        
    except Exception as e:
        print(f"Error en comorbilidades: {e}")
        return render(request, 'comorbilidades.html', {
            'datos_comorbilidades': [],
            'datos_complicaciones': [],
            'grafico_comorbilidades': None,
            'grafico_complicaciones': None,
            'error': str(e),
            'titulo': 'Comorbilidades y Complicaciones DM2'
        })

def prevalencia_zona(request):
    try:
        # Obtener datos de la base de datos y ordenar por prevalencia numérica
        datos_raw = Dm2Departamentos.objects.all()
        
        # Ordenar por prevalencia convertida a número
        def get_prevalencia_numerica(registro):
            if registro.prevalencia:
                try:
                    return float(str(registro.prevalencia).replace('%', '').strip())
                except:
                    return 0
            return 0
        
        datos_raw_ordenados = sorted(datos_raw, key=get_prevalencia_numerica, reverse=True)
        
        # Crear datos con nombres
        datos = []
        for registro in datos_raw_ordenados:
            # Crear objeto
            class ZonaLimpia:
                def __init__(self, original):
                    # Limpiar el nombre
                    zona_original = str(original.zona).strip() if original.zona else ""
                    self.zona = zona_original.replace('ZONA ', '') if zona_original.startswith('ZONA ') else zona_original
                    self.zona_original = zona_original  # Guardar original para el mapa
                    self.pacientes = original.pacientes
                    self.prevalencia = original.prevalencia
            
            datos.append(ZonaLimpia(registro))
        
        # Crear mapa con puntos
        mapa_html = None
        if datos:
            # Coordenadas de las 18 zonas de salud de Valencia
            coordenadas_zonas = {
                'ZONA TRAFALGAR (VALENCIA)': (39.4614, -0.3452),  # Camins al Grau
                'ZONA BENIMACLET (VALENCIA)': (39.4867, -0.3585), # Benimaclet
                'ZONA CHILE': (39.4709, -0.3556),                 # Russafa
                'ZONA NAZARET (VALENCIA)': (39.4505, -0.3297),    # Nazaret 
                'ZONA MALVA': (39.4765, -0.3254),                 # Malva-rosa/Cabañal
                'ZONA FOIOS': (39.5354, -0.3524),                 # Foios (norte)
                'ZONA MASSAMAGRELL': (39.5709, -0.3348),          # Massamagrell (norte)
                'ZONA SERRERIA 1 (VALENCIA)': (39.4688, -0.3349), # Serrería
                'ZONA TAVERNES BLANQUES': (39.5070, -0.3623),     # Tavernes Blanques
                'ZONA SERRERIA 2 (VALENCIA)': (39.4685, -0.3361), # Serrería 2
                'ZONA ALMASSERA': (39.5123, -0.3587),             # Almàssera
                'ZONA RAFELBUNYOL': (39.5863, -0.3342),           # Rafelbunyol
                'ZONA SALVADOR PAU (VALENCIA)': (39.4703, -0.3516), # Salvador Pau
                'ZONA MELIANA': (39.5291, -0.3475),               # Meliana
                'ZONA MUSEROS': (39.5632, -0.3432),               # Museros
                'ZONA ALFAHUIR': (39.4923, -0.3590),              # Alfahuir
                'ZONA REPUBLICA ARGENTINA (VALENCIA)': (39.4718, -0.3520), # República Argentina
                'ZONA ALBORAIA': (39.4943, -0.3502)               # Alboraia
            }
            
            lats, lons, prevalencias, nombres, pacientes = [], [], [], [], []
            
            for registro in datos:
                zona_bd = registro.zona_original
                
                if zona_bd in coordenadas_zonas:
                    lat, lon = coordenadas_zonas[zona_bd]
                    lats.append(lat)
                    lons.append(lon)
                    nombres.append(registro.zona)
                    
                    # Convertir prevalencia a número
                    if registro.prevalencia:
                        prev_str = str(registro.prevalencia).replace('%', '').strip()
                        try:
                            prev_num = float(prev_str)
                            prevalencias.append(prev_num)
                        except:
                            prevalencias.append(0)
                    else:
                        prevalencias.append(0)
                    
                    pacientes.append(registro.pacientes if registro.pacientes else 0)
            
            if lats and prevalencias:
                # Crear mapa de puntos con tamaño y color según prevalencia
                fig = go.Figure(go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode='markers',
                    marker=dict(
                        size=[max(20, min(50, p*3.5)) for p in prevalencias],  # Tamaño controlado entre 20-50
                        color=prevalencias,
                        colorscale=[
                            [0, '#22C55E'],
                            [0.33, '#EAB308'],
                            [0.66, '#F97316'],
                            [1, '#DC2626']
                        ],
                        showscale=True,
                        colorbar=dict(
                            title=dict(
                                text="Prevalencia (%)",
                                font=dict(size=12)
                            ),
                            len=0.4,
                            thickness=15,
                            x=0.88,         
                            y=0.05,
                            xanchor="right",
                            yanchor="bottom",
                            bgcolor="rgba(255,255,255,0.9)",  
                            bordercolor="rgba(0,0,0,0.4)",
                            borderwidth=1,
                            tickfont=dict(size=10)
                        ),
                        sizemin=20,
                        opacity=0.8
                    ),
                    text=nombres,
                    customdata=list(zip(prevalencias, pacientes)),
                    hovertemplate="<b>%{text}</b><br>" +
                                "Prevalencia: %{customdata[0]:.2f}%<br>" +
                                "Pacientes: %{customdata[1]}<br>" +
                                "<extra></extra>",
                    hoverlabel=dict(
                        bgcolor="white",
                        bordercolor="black",
                        font_size=12
                    )
                ))
                
                # layout
                fig.update_layout(
                    mapbox_style="open-street-map",
                    mapbox=dict(
                        center=dict(lat=39.5200, lon=-0.3450),
                        zoom=12.5,
                        bounds=dict(
                            west=-0.48,   
                            east=-0.22,   
                            south=39.35,  
                            north=39.62  
                        ),
                        accesstoken=None,
                    ),
                    title={
                        'text': f'Prevalencia de DM2 por Zonas de Salud - Valencia ({len(lats)}/18 zonas)',
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 18, 'color': '#2c3e50', 'family': 'Arial'}
                    },
                    height=650,
                    width=1200,   
                    margin=dict(r=80, t=80, l=20, b=20), 
                    paper_bgcolor='white',
                    showlegend=False
                )
                
                mapa_html = opy.plot(
                    fig, 
                    output_type='div', 
                    include_plotlyjs=True
                )
                
                print(f"Mapa de puntos generado con {len(lats)} zonas")
                print(f"Rango de prevalencias: {min(prevalencias):.2f}% - {max(prevalencias):.2f}%")
            else:
                mapa_html = '''
                <div class="alert alert-warning">
                    <h5>No hay datos suficientes para el mapa</h5>
                    <p>No se pudieron encontrar coordenadas para las zonas de salud.</p>
                </div>
                '''
        
        return render(request, 'prevalencia_zona.html', {
            'datos': datos,
            'titulo': 'Prevalencia DM2 por Zona',
            'mapa': mapa_html
        })
        
    except Exception as e:
        print(f"Error en prevalencia_zona: {e}")
        import traceback
        traceback.print_exc()
        return render(request, 'prevalencia_zona.html', {
            'datos': [],
            'error': str(e),
            'titulo': 'Prevalencia DM2 por Zona'
        })

def costes_totales(request):
    try:
        registro_costes = Dm2CostesTotales.objects.first()
        
        if not registro_costes:
            return render(request, 'costes_totales.html', {
                'datos': [],
                'error': 'No se encontraron datos de costes',
                'titulo': 'Análisis de Costes Totales'
            })
        
        # Crear estructura de datos para el gráfico y tabla
        conceptos_costes = [
            ('CATP', float(registro_costes.ct_catp or 0)),
            ('CEX', float(registro_costes.ct_cex or 0)),
            ('Urgencias', float(registro_costes.ct_urgencias or 0)),
            ('Ingresos', float(registro_costes.ct_ingresos or 0)),
            ('Farmacéutico', float(registro_costes.ct_importe_farmaceutico or 0)),
            ('Total General', float(registro_costes.coste_total or 0))
        ]
        
        # Filtro
        datos_grafico = [(concepto, coste) for concepto, coste in conceptos_costes if concepto != 'Total General' and coste > 0]
        datos_grafico.sort(key=lambda x: x[1], reverse=True)
        
        # Crear gráfico de barras si hay datos
        grafico_html = None
        if datos_grafico:
            conceptos = [d[0] for d in datos_grafico]
            costes = [d[1] for d in datos_grafico]
            
            # Crear gráfico de barras horizontales
            fig = go.Figure(go.Bar(
                x=costes,
                y=conceptos,
                orientation='h',
                marker=dict(
                    color=costes,
                    colorscale=[
                        [0, '#E8F6F3'],      
                        [0.25, '#A8DBA8'],  
                        [0.5, '#79B79A'],    
                        [0.75, '#4A90A4'], 
                        [1, '#C0392B']       
                    ],
                    showscale=False,
                    line=dict(color='rgba(255,255,255,0.8)', width=1)
                ),
                text=[f'{c:,.0f}€' for c in costes],
                textposition='outside',
                textfont=dict(size=11, color='#2c3e50'),
                hovertemplate="<b>%{y}</b><br>" +
                            "Coste: %{x:,.0f}€<br>" +
                            "<extra></extra>"
            ))
            
            #layout
            fig.update_layout(
                title={
                    'text': 'Distribución de Costes Totales por Concepto',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16, 'color': '#2c3e50'}
                },
                xaxis_title='Coste (€)',
                yaxis_title='Concepto',
                height=max(400, len(datos_grafico) * 60),
                width=1000,
                showlegend=False,
                #ejes
                xaxis=dict(
                    title_font=dict(size=12),
                    tickfont=dict(size=11),
                    tickformat=',.0f',
                    showgrid=True,
                    gridcolor='lightgray',
                    gridwidth=0.5
                ),
                yaxis=dict(
                    title_font=dict(size=12),
                    tickfont=dict(size=11),
                    automargin=True
                ),
                paper_bgcolor='white',
                plot_bgcolor='white',
                margin=dict(
                    l=100,
                    r=150,
                    t=80,
                    b=60
                )
            )
            
            grafico_html = opy.plot(
                fig, 
                output_type='div', 
                include_plotlyjs=True
            )
        
        #estadísticas
        total_coste = float(registro_costes.coste_total or 0)
        costes_parciales = [coste for concepto, coste in datos_grafico]
        promedio_coste = sum(costes_parciales) / len(costes_parciales) if costes_parciales else 0
        
        #porcentajes
        datos_tabla = []
        for concepto, coste in conceptos_costes:
            porcentaje = (coste / total_coste * 100) if total_coste > 0 and coste > 0 else 0
            datos_tabla.append({
                'concepto': concepto,
                'coste': coste,
                'porcentaje': porcentaje
            })
        
        return render(request, 'costes_totales.html', {
            'datos': datos_tabla,
            'titulo': 'Análisis de Costes Totales',
            'grafico': grafico_html,
            'total_coste': total_coste,
            'promedio_coste': promedio_coste,
            'num_conceptos': len(datos_grafico)
        })
        
    except Exception as e:
        print(f"Error en costes_totales: {e}")
        import traceback
        traceback.print_exc()
        return render(request, 'costes_totales.html', {
            'datos': [],
            'error': str(e),
            'titulo': 'Análisis de Costes Totales'
        })



def prediccion_gasto(request):
    try:
        pacientes = PredecirFarmacia.objects.filter(
            fecha_nacimiento__isnull=False,
            sexo__isnull=False,
            coste_farmaceutico__isnull=False  
        )
        
        if len(pacientes) < 50:
            return render(request, 'prediccion_gasto.html', {
                'error': 'No hay suficientes datos para entrenar el modelo',
                'titulo': 'Predicción Gasto Farmacéutico DM2'
            })
        
        # Convertir a DataFrame
        data = []
        for paciente in pacientes:
            if paciente.edad is not None:
                data.append({
                    'edad': paciente.edad,
                    'sexo': paciente.sexo,
                    'coste_farmaceutico': float(paciente.coste_farmaceutico),
                    'anyos_con_dm2': paciente.anyos_con_dm2 or 0,
                    'num_comorbilidades': paciente.num_comorbilidades or 0,
                    'num_complicaciones': paciente.num_complicaciones or 0,
                    'gravedad_1': paciente.gravedad_1 or 0,
                    'gravedad_2': paciente.gravedad_2 or 0,
                    'gravedad_3': paciente.gravedad_3 or 0,
                    'epoc': paciente.epoc or 0,
                    'asma_epoc': paciente.asma_epoc or 0,
                    'anemias': paciente.anemias or 0,
                    'tabaco': paciente.tabaco or 0,
                    'osteoporosis': paciente.osteoporosis or 0,
                    'insuficiencia_venosa': paciente.insuficiencia_venosa or 0,
                    'hta': paciente.hta or 0,
                    'dislipemias': paciente.dislipemias or 0,
                    'obesidad': paciente.obesidad or 0,
                    'erc': paciente.erc or 0,
                    'ic': paciente.ic or 0,
                    'ansiedad': paciente.ansiedad or 0,
                    'insomnio': paciente.insomnio or 0,
                    'depresion': paciente.depresion or 0,
                    'demencia': paciente.demencia or 0,
                    'alzheimer': paciente.alzheimer or 0,
                    'fa': paciente.fa or 0,
                    'artrosis': paciente.artrosis or 0,
                    'cardiopatia_isquemica': paciente.cardiopatia_isquemica or 0,
                    'enf_arterial_periferica': paciente.enf_arterial_periferica or 0,
                    'enf_cerebrovascular': paciente.enf_cerebrovascular or 0,
                    'nefropatia_diabetica': paciente.nefropatia_diabetica or 0,
                    'retinopatia_diabetica': paciente.retinopatia_diabetica or 0,
                    'neuropatia_diabetica': paciente.neuropatia_diabetica or 0,
                    'c_microvasculares': paciente.c_microvasculares or 0,
                    'c_macrovasculares': paciente.c_macrovasculares or 0,
                })
        
        df = pd.DataFrame(data)
        
        # Codificar sexo
        df['sexo_num'] = df['sexo'].map({'M': 1, 'F': 0, 'H': 1})
        df = df.dropna()
        
        # Definir features
        feature_columns = [
            'edad', 'sexo_num',
            # NUEVAS VARIABLES PREDICTIVAS
            'anyos_con_dm2', 'num_comorbilidades', 'num_complicaciones', 'gravedad_1', 'gravedad_2', 'gravedad_3',
            # Comorbilidades
            'epoc', 'asma_epoc', 'anemias', 'tabaco', 'osteoporosis', 
            'insuficiencia_venosa', 'hta', 'dislipemias', 'obesidad', 
            'erc', 'ic', 'ansiedad', 'insomnio', 'depresion', 'demencia', 
            'alzheimer', 'fa', 'artrosis',
            # Complicaciones
            'cardiopatia_isquemica', 'enf_arterial_periferica', 
            'enf_cerebrovascular', 'nefropatia_diabetica', 
            'retinopatia_diabetica', 'neuropatia_diabetica', 
            'c_microvasculares', 'c_macrovasculares'
        ]
        
        # Preparar datos para ML
        X = df[feature_columns]
        y = df['coste_farmaceutico']
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entrenar modelos
        modelo_lineal = LinearRegression()
        modelo_lineal.fit(X_train, y_train)
        
        modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
        modelo_rf.fit(X_train, y_train)
        
        # Predicciones
        y_pred_lineal = modelo_lineal.predict(X_test)
        y_pred_rf = modelo_rf.predict(X_test)
        
        # Métricas
        mse_lineal = mean_squared_error(y_test, y_pred_lineal)
        r2_lineal = r2_score(y_test, y_pred_lineal)
        
        mse_rf = mean_squared_error(y_test, y_pred_rf)
        r2_rf = r2_score(y_test, y_pred_rf)
        
        # Importancia de features (Random Forest)
        feature_importance = dict(zip(feature_columns, modelo_rf.feature_importances_))
        # Ordenar por importancia
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))        
        top_features = list(feature_importance.keys())[:8]  # Top 8 features más importantes
        modelo_serializado = base64.b64encode(pickle.dumps(modelo_rf)).decode('utf-8')
        request.session['modelo_rf_farmacia'] = modelo_serializado
        request.session['feature_columns_farmacia'] = feature_columns
        request.session['top_features_farmacia'] = top_features
        
        # Estadísticas adicionales
        stats = {
            'total_pacientes': len(df),
            'gasto_promedio': float(df['coste_farmaceutico'].mean()),    
            'gasto_mediano': float(df['coste_farmaceutico'].median()),   
            'gasto_max': float(df['coste_farmaceutico'].max()),          
            'gasto_min': float(df['coste_farmaceutico'].min()),          
            'edad_promedio': float(df['edad'].mean()),
            
            # Métricas modelo lineal
            'r2_lineal': float(r2_lineal),
            'mse_lineal': float(mse_lineal),
            
            # Métricas Random Forest
            'r2_rf': float(r2_rf),
            'mse_rf': float(mse_rf),
            
            # Features más importantes (top 10)
            'features_importantes': {k: float(v) for k, v in list(feature_importance.items())[:10]},
            
            # Prevalencia de comorbilidades principales
            'prevalencias': {
                'hta': float(df['hta'].mean()),
                'dislipemias': float(df['dislipemias'].mean()),
                'obesidad': float(df['obesidad'].mean()),
                'ic': float(df['ic'].mean()),
                'erc': float(df['erc'].mean()),
                'nefropatia_diabetica': float(df['nefropatia_diabetica'].mean()),
                'retinopatia_diabetica': float(df['retinopatia_diabetica'].mean()),
                'c_macrovasculares': float(df['c_macrovasculares'].mean()),
                'c_microvasculares': float(df['c_microvasculares'].mean()),
            }
        }
        
        return render(request, 'prediccion_gasto.html', {
            'stats': stats,
            'titulo': 'Predicción Gasto Farmacéutico DM2',
            'modelo_usado': 'Regresión Lineal y Random Forest',
            'top_features': top_features
        })
        
    except Exception as e:
        print(f"Error en prediccion_gasto: {e}")
        return render(request, 'prediccion_gasto.html', {
            'error': str(e),
            'titulo': 'Predicción Gasto Farmacéutico DM2'
        })

def prediccion_coste_total(request):
    try:
        pacientes = PredecirCosteTotal.objects.filter(
            fecha_nacimiento__isnull=False,
            sexo__isnull=False,
            coste_total_paciente__isnull=False,
            coste_total_paciente__gt=0 
        )
        
        if len(pacientes) < 50:
            return render(request, 'prediccion_coste_total.html', {
                'error': 'No hay suficientes datos para entrenar el modelo',
                'titulo': 'Predicción Coste Total DM2'
            })
        
        # Convertir a DataFrame
        data = []
        for paciente in pacientes:
            if paciente.edad is not None:
                data.append({
                    'edad': paciente.edad,
                    'sexo': paciente.sexo,
                    'coste_total_paciente': float(paciente.coste_total_paciente),
                    'anyos_con_dm2': paciente.anyos_con_dm2 or 0,
                    'num_comorbilidades': paciente.num_comorbilidades or 0,
                    'num_complicaciones': paciente.num_complicaciones or 0,
                    'gravedad_1': paciente.gravedad_1 or 0,
                    'gravedad_2': paciente.gravedad_2 or 0,
                    'gravedad_3': paciente.gravedad_3 or 0,
                    'coste_farmaceutico': float(paciente.coste_farmaceutico or 0),
                    'coste_ambulatorio': float(paciente.coste_ambulatorio),
                    'coste_hospitalario': float(paciente.coste_hospitalario),
                    'coste_catp_cen': float(paciente.coste_catp_cen or 0),
                    'coste_catp_domi': float(paciente.coste_catp_domi or 0),
                    'coste_urgencias': float(paciente.coste_urgencias or 0),
                    'coste_ingresos': float(paciente.coste_ingresos or 0),
                    'epoc': paciente.epoc or 0,
                    'asma_epoc': paciente.asma_epoc or 0,
                    'anemias': paciente.anemias or 0,
                    'tabaco': paciente.tabaco or 0,
                    'osteoporosis': paciente.osteoporosis or 0,
                    'insuficiencia_venosa': paciente.insuficiencia_venosa or 0,
                    'hta': paciente.hta or 0,
                    'dislipemias': paciente.dislipemias or 0,
                    'obesidad': paciente.obesidad or 0,
                    'erc': paciente.erc or 0,
                    'ic': paciente.ic or 0,
                    'ansiedad': paciente.ansiedad or 0,
                    'insomnio': paciente.insomnio or 0,
                    'depresion': paciente.depresion or 0,
                    'demencia': paciente.demencia or 0,
                    'alzheimer': paciente.alzheimer or 0,
                    'fa': paciente.fa or 0,
                    'artrosis': paciente.artrosis or 0,
                    'cardiopatia_isquemica': paciente.cardiopatia_isquemica or 0,
                    'enf_arterial_periferica': paciente.enf_arterial_periferica or 0,
                    'enf_cerebrovascular': paciente.enf_cerebrovascular or 0,
                    'nefropatia_diabetica': paciente.nefropatia_diabetica or 0,
                    'retinopatia_diabetica': paciente.retinopatia_diabetica or 0,
                    'neuropatia_diabetica': paciente.neuropatia_diabetica or 0,
                    'c_microvasculares': paciente.c_microvasculares or 0,
                    'c_macrovasculares': paciente.c_macrovasculares or 0,
                })
        
        df = pd.DataFrame(data)
        
        # Codificarr sexo
        df['sexo_num'] = df['sexo'].map({'M': 1, 'F': 0, 'H': 1})
        df = df.dropna()
        
        # Definir features
        feature_columns = [
            'edad', 'sexo_num',
            'anyos_con_dm2', 'num_comorbilidades', 'num_complicaciones', 'gravedad_1', 'gravedad_2', 'gravedad_3',
            'epoc', 'asma_epoc', 'anemias', 'tabaco', 'osteoporosis', 
            'insuficiencia_venosa', 'hta', 'dislipemias', 'obesidad', 
            'erc', 'ic', 'ansiedad', 'insomnio', 'depresion', 'demencia', 
            'alzheimer', 'fa', 'artrosis',
            'cardiopatia_isquemica', 'enf_arterial_periferica', 
            'enf_cerebrovascular', 'nefropatia_diabetica', 
            'retinopatia_diabetica', 'neuropatia_diabetica', 
            'c_microvasculares', 'c_macrovasculares'
        ]
        
        # Preparar datos para ML
        X = df[feature_columns]
        y = df['coste_total_paciente']
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entrenar modelos
        modelo_lineal = LinearRegression()
        modelo_lineal.fit(X_train, y_train)
        
        modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
        modelo_rf.fit(X_train, y_train)
        
        # Predicciones
        y_pred_lineal = modelo_lineal.predict(X_test)
        y_pred_rf = modelo_rf.predict(X_test)
        
        # Métricas
        mse_lineal = mean_squared_error(y_test, y_pred_lineal)
        r2_lineal = r2_score(y_test, y_pred_lineal)
        
        mse_rf = mean_squared_error(y_test, y_pred_rf)
        r2_rf = r2_score(y_test, y_pred_rf)
        
        # Importancia de features (Random Forest)
        feature_importance = dict(zip(feature_columns, modelo_rf.feature_importances_))
        # Ordenar por importancia
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        top_features = list(feature_importance.keys())[:8]  # Top 8 features más importantes
        modelo_serializado = base64.b64encode(pickle.dumps(modelo_rf)).decode('utf-8')
        request.session['modelo_rf_coste'] = modelo_serializado
        request.session['feature_columns_coste'] = feature_columns
        request.session['top_features_coste'] = top_features
        
        # Análisis de distribución de costes
        promedios_costes = {
            'farmaceutico': float(df['coste_farmaceutico'].mean()),
            'ambulatorio': float(df['coste_ambulatorio'].mean()),
            'hospitalario': float(df['coste_hospitalario'].mean()),
            'catp_centro': float(df['coste_catp_cen'].mean()),
            'catp_domicilio': float(df['coste_catp_domi'].mean()),
            'urgencias': float(df['coste_urgencias'].mean()),
            'ingresos': float(df['coste_ingresos'].mean()),
        }
        
        # Porcentajes de cada concepto sobre el total
        porcentajes_costes = {}
        total_promedio = float(df['coste_total_paciente'].mean())
        for concepto, promedio in promedios_costes.items():
            porcentajes_costes[concepto] = (promedio / total_promedio * 100) if total_promedio > 0 else 0
        
        # Estadísticas adicionales
        stats = {
            'total_pacientes': len(df),
            'coste_total_promedio': float(df['coste_total_paciente'].mean()),
            'coste_total_mediano': float(df['coste_total_paciente'].median()),
            'coste_total_max': float(df['coste_total_paciente'].max()),
            'coste_total_min': float(df['coste_total_paciente'].min()),
            'edad_promedio': float(df['edad'].mean()),
            
            # Métricas modelo lineal
            'r2_lineal': float(r2_lineal),
            'mse_lineal': float(mse_lineal),
            
            # Métricas Random Forest
            'r2_rf': float(r2_rf),
            'mse_rf': float(mse_rf),
            
            # Features más importantes (top 10)
            'features_importantes': {k: float(v) for k, v in list(feature_importance.items())[:10]},
            
            # Distribución de costes
            'promedios_costes': promedios_costes,
            'porcentajes_costes': porcentajes_costes,
            
            # Prevalencia de comorbilidades principales
            'prevalencias': {
                'hta': float(df['hta'].mean()),
                'dislipemias': float(df['dislipemias'].mean()),
                'obesidad': float(df['obesidad'].mean()),
                'ic': float(df['ic'].mean()),
                'erc': float(df['erc'].mean()),
                'nefropatia_diabetica': float(df['nefropatia_diabetica'].mean()),
                'retinopatia_diabetica': float(df['retinopatia_diabetica'].mean()),
                'c_macrovasculares': float(df['c_macrovasculares'].mean()),
                'c_microvasculares': float(df['c_microvasculares'].mean()),
            }
        }
        
        return render(request, 'prediccion_coste_total.html', {
            'stats': stats,
            'titulo': 'Predicción Coste Total DM2',
            'modelo_usado': 'Regresión Lineal y Random Forest',
            'top_features': top_features
        })
        
    except Exception as e:
        print(f"Error en prediccion_coste_total: {e}")
        return render(request, 'prediccion_coste_total.html', {
            'error': str(e),
            'titulo': 'Predicción Coste Total DM2'
        })

def prediccion_individual_farmacia(request):
    if request.method == 'POST':
        try:
            modelo_serializado = request.session.get('modelo_rf_farmacia')
            if modelo_serializado:
                modelo_rf = pickle.loads(base64.b64decode(modelo_serializado.encode('utf-8')))
            else:
                modelo_rf = None
            feature_columns = request.session.get('feature_columns_farmacia')
            
            if not modelo_rf or not feature_columns:
                return JsonResponse({'error': 'Modelo no disponible. Vuelve a la pestaña Resumen primero.'})
            
            datos_paciente = {}
            
            for col in feature_columns:
                datos_paciente[col] = 0
            
            datos_paciente['edad'] = int(request.POST.get('edad', 65))
            datos_paciente['sexo_num'] = 1 if request.POST.get('sexo') == 'M' else 0
            datos_paciente['anyos_con_dm2'] = int(request.POST.get('anyos_con_dm2', 0))
            datos_paciente['num_comorbilidades'] = int(request.POST.get('num_comorbilidades', 0))
            datos_paciente['num_complicaciones'] = int(request.POST.get('num_complicaciones', 0))
            ing_num = int(request.POST.get('ing_total_num', 0))
            datos_paciente['gravedad_1'] = 1 if ing_num == 1 else 0
            datos_paciente['gravedad_2'] = 1 if ing_num == 2 else 0
            datos_paciente['gravedad_3'] = 1 if ing_num >= 3 else 0
            
            comorbilidades = ['hta', 'dislipemias', 'obesidad', 'ic', 'erc', 'ansiedad', 'depresion']
            for comorbilidad in comorbilidades:
                if comorbilidad in feature_columns:
                    datos_paciente[comorbilidad] = 1 if request.POST.get(comorbilidad) else 0
            
            complicaciones = ['nefropatia_diabetica', 'retinopatia_diabetica', 'cardiopatia_isquemica', 'c_macrovasculares', 'c_microvasculares']
            for complicacion in complicaciones:
                if complicacion in feature_columns:
                    datos_paciente[complicacion] = 1 if request.POST.get(complicacion) else 0
            
            # Crear DataFrame para predicción
            df_prediccion = pd.DataFrame([datos_paciente])
            df_prediccion = df_prediccion[feature_columns]  # Asegurar orden correcto
            
            # Hacer predicción
            prediccion = modelo_rf.predict(df_prediccion)[0]
            
            return JsonResponse({
                'success': True,
                'prediccion': round(prediccion, 2),
                'mensaje': f'Coste farmacéutico estimado: {prediccion:.2f}€'
            })
            
        except Exception as e:
            return JsonResponse({'error': f'Error en predicción: {str(e)}'})
    
    return JsonResponse({'error': 'Método no permitido'})

def prediccion_individual_coste_total(request):
    if request.method == 'POST':
        try:
            # Obtener modelo
            modelo_serializado = request.session.get('modelo_rf_coste')
            if modelo_serializado:
                modelo_rf = pickle.loads(base64.b64decode(modelo_serializado.encode('utf-8')))
            else:
                modelo_rf = None
            feature_columns = request.session.get('feature_columns_coste')
            
            if not modelo_rf or not feature_columns:
                return JsonResponse({'error': 'Modelo no disponible. Vuelve a la pestaña Resumen primero.'})
            
            datos_paciente = {}
            
            for col in feature_columns:
                datos_paciente[col] = 0
            
            datos_paciente['edad'] = int(request.POST.get('edad', 65))
            datos_paciente['sexo_num'] = 1 if request.POST.get('sexo') == 'M' else 0
            datos_paciente['anyos_con_dm2'] = int(request.POST.get('anyos_con_dm2', 0))
            datos_paciente['num_comorbilidades'] = int(request.POST.get('num_comorbilidades', 0))
            datos_paciente['num_complicaciones'] = int(request.POST.get('num_complicaciones', 0))
            ing_num = int(request.POST.get('ing_total_num', 0))
            datos_paciente['gravedad_1'] = 1 if ing_num == 1 else 0
            datos_paciente['gravedad_2'] = 1 if ing_num == 2 else 0
            datos_paciente['gravedad_3'] = 1 if ing_num >= 3 else 0
            
            comorbilidades = ['hta', 'dislipemias', 'obesidad', 'ic', 'erc', 'ansiedad', 'depresion']
            for comorbilidad in comorbilidades:
                if comorbilidad in feature_columns:
                    datos_paciente[comorbilidad] = 1 if request.POST.get(comorbilidad) else 0
            
            complicaciones = ['nefropatia_diabetica', 'retinopatia_diabetica', 'cardiopatia_isquemica', 'c_macrovasculares', 'c_microvasculares']
            for complicacion in complicaciones:
                if complicacion in feature_columns:
                    datos_paciente[complicacion] = 1 if request.POST.get(complicacion) else 0
            
            # Crear DataFrame para predicción
            df_prediccion = pd.DataFrame([datos_paciente])
            df_prediccion = df_prediccion[feature_columns]  # Asegurar orden correcto
            
            # Hacer predicción
            prediccion = modelo_rf.predict(df_prediccion)[0]
            
            return JsonResponse({
                'success': True,
                'prediccion': round(prediccion, 2),
                'mensaje': f'Coste total estimado: {prediccion:.2f}€'
            })
            
        except Exception as e:
            return JsonResponse({'error': f'Error en predicción: {str(e)}'})
    
    return JsonResponse({'error': 'Método no permitido'})
