#!/usr/bin/env python3
"""
Script para generar SOLO las tablas de coeficientes y p-valores
de los modelos de regresión lineal para DM2
"""

import os
import sys
import django
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configurar Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'miweb.settings')
django.setup()

from dashboard.models import PredecirFarmacia, PredecirCosteTotal

def generar_tabla_coeficientes(modelo, X_train, y_train, y_test, y_pred, features, titulo):
    """Generar tabla visual de coeficientes y p-valores"""
    try:
        # Asegurar que todos los arrays sean float
        X_train = np.array(X_train, dtype=float)
        y_train = np.array(y_train, dtype=float)
        y_test = np.array(y_test, dtype=float)
        y_pred = np.array(y_pred, dtype=float)
        
        n = len(X_train)
        p = len(features)
        
        # Calcular estadísticas correctamente
        residuos = y_test - y_pred
        mse = np.sum(residuos ** 2) / (n - p - 1)
        
        # CORRECCIÓN: Usar datos de entrenamiento para matriz de covarianza
        X_design = np.column_stack([np.ones(len(X_train)), X_train])
        
        # Calcular p-valores correctamente
        try:
            # Matriz (X'X)^-1
            XTX = X_design.T @ X_design
            XTX_inv = np.linalg.inv(XTX)
            
            # Varianza de los coeficientes
            var_coeff = mse * np.diag(XTX_inv)
            std_errors = np.sqrt(var_coeff)
            
            # Coeficientes con intercepto - conversión explícita a float
            intercepto = float(modelo.intercept_)
            coeficientes = np.array(modelo.coef_, dtype=float)
            coefs_with_intercept = np.append(intercepto, coeficientes)
            
            # t-statistics y p-valores
            t_stats = coefs_with_intercept / std_errors
            grados_libertad = n - p - 1
            p_valores = 2 * (1 - stats.t.cdf(np.abs(t_stats), grados_libertad))
            
            print(f"✅ Cálculo exitoso - DOF: {grados_libertad}, MSE: {mse:.2f}")
            
        except np.linalg.LinAlgError as e:
            print(f"⚠️ Matriz singular - usando regularización: {e}")
            
            # Agregar pequeña regularización a la diagonal
            regularization = 1e-6
            XTX_reg = XTX + regularization * np.eye(XTX.shape[0])
            
            try:
                XTX_inv = np.linalg.inv(XTX_reg)
                var_coeff = mse * np.diag(XTX_inv)
                std_errors = np.sqrt(np.abs(var_coeff))  # abs por seguridad
                
                intercepto = float(modelo.intercept_)
                coeficientes = np.array(modelo.coef_, dtype=float)
                coefs_with_intercept = np.append(intercepto, coeficientes)
                
                # Evitar división por cero
                std_errors = np.where(std_errors < 1e-10, 1e-10, std_errors)
                
                t_stats = coefs_with_intercept / std_errors
                grados_libertad = max(n - p - 1, 1)
                p_valores = 2 * (1 - stats.t.cdf(np.abs(t_stats), grados_libertad))
                
                print(f"✅ Regularización exitosa - DOF: {grados_libertad}")
                
            except Exception as e2:
                print(f"❌ Error en regularización: {e2}")
                # Fallback: p-valores aproximados basados en normalidad
                intercepto = float(modelo.intercept_)
                coeficientes = np.array(modelo.coef_, dtype=float)
                coefs_with_intercept = np.append(intercepto, coeficientes)
                
                # Estimación rugosa de std errors basada en los residuos
                residual_std = np.std(residuos)
                std_errors = np.ones(len(coefs_with_intercept)) * residual_std / np.sqrt(n)
                
                z_stats = coefs_with_intercept / std_errors
                p_valores = 2 * (1 - stats.norm.cdf(np.abs(z_stats)))  # Aproximación normal
                
                print(f"✅ Usando aproximación normal")
        
        except Exception as e:
            print(f"❌ Error general en cálculo: {e}")
            return None
        
        # Preparar datos para la tabla
        variables = ['Intercepto'] + features
        coeficientes = coefs_with_intercept
        significancia = []
        
        # Asegurar que p_valores sean válidos
        p_valores = np.clip(p_valores, 0, 1)  # Limitar entre 0 y 1
        
        for p_val in p_valores:
            if p_val < 0.001:
                significancia.append('***')
            elif p_val < 0.01:
                significancia.append('**')
            elif p_val < 0.05:
                significancia.append('*')
            else:
                significancia.append('')
        
        # Debug info
        print(f"📊 Resumen estadístico {titulo}:")
        print(f"   N observaciones: {n}")
        print(f"   Grados libertad: {n-p-1}")
        print(f"   MSE: {mse:.2f}")
        print(f"   P-valores rango: {np.min(p_valores):.6f} - {np.max(p_valores):.6f}")
        significant_vars = sum([1 for p in p_valores if p < 0.05])
        print(f"   Variables significativas (p<0.05): {significant_vars}/{len(variables)}")
        
        # Crear la figura
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('tight')
        ax.axis('off')
        
        # Título
        fig.suptitle(f'Coeficientes y P-valores - Regresión Lineal {titulo}', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        # Preparar datos de la tabla
        tabla_datos = []
        for i, (var, coef, p_val, sig) in enumerate(zip(variables, coeficientes, p_valores, significancia)):
            tabla_datos.append([
                var, 
                f"{coef:.6f}", 
                f"{p_val:.6f}", 
                sig,
                f"{abs(coef):.6f}"  # Para ordenar por importancia
            ])
        
        # Ordenar por valor absoluto del coeficiente (más importantes primero)
        tabla_datos.sort(key=lambda x: float(x[4]), reverse=True)
        
        # Preparar tabla final (sin la columna de ordenamiento)
        tabla_final = []
        colores = []
        for row in tabla_datos:  # Mostrar todas las variables
            tabla_final.append(row[:4])  # Sin la columna de ordenamiento
            
            # Colorear según significancia
            p_val = float(row[2])
            if p_val < 0.001:
                colores.append(['#90EE90'] * 4)  # Verde claro
            elif p_val < 0.01:
                colores.append(['#ADD8E6'] * 4)  # Azul claro
            elif p_val < 0.05:
                colores.append(['#FFFFE0'] * 4)  # Amarillo claro
            else:
                colores.append(['#FFFFFF'] * 4)  # Blanco
        
        # Crear la tabla
        tabla = ax.table(cellText=tabla_final,
                        colLabels=['Variable', 'Coeficiente', 'P-valor', 'Significancia'],
                        cellColours=colores,
                        cellLoc='center',
                        loc='center')
        
        # Formatear tabla
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(10)
        tabla.scale(1.2, 1.8)
        
        # Estilo del encabezado
        for i in range(4):
            tabla[(0, i)].set_facecolor('#2E8B57')  # Verde oscuro
            tabla[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)
        
        # Agregar información estadística
        stats_text = f"""
Modelo: Regresión Lineal - {titulo}
Variables predictoras: {len(features)}
Muestra de entrenamiento: {n} observaciones

Significancia estadística:
*** p < 0.001 (Altamente significativo)
**  p < 0.01  (Muy significativo) 
*   p < 0.05  (Significativo)

Interpretación de coeficientes:
• Valores positivos: aumentan la variable dependiente
• Valores negativos: disminuyen la variable dependiente
• Ordenado por magnitud del impacto (valor absoluto)
        """
        
        ax.text(0.5, -0.25, stats_text, transform=ax.transAxes, 
                fontsize=9, ha='center', va='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F8FF", alpha=0.8))
        
        plt.tight_layout()
        filename = f'tabla_coeficientes_{titulo.lower().replace(" ", "_").replace("á", "a").replace("é", "e")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Tabla generada: {filename}")
        plt.show()
        
        return filename
        
    except Exception as e:
        print(f"❌ Error generando tabla de coeficientes para {titulo}: {e}")
        return None

def preparar_y_analizar_farmacia():
    """Preparar datos y generar tabla para predicción farmacéutica"""
    print("\n" + "="*60)
    print("GENERANDO TABLA: PREDICCIÓN GASTO FARMACÉUTICO")
    print("="*60)
    
    # Cargar datos
    queryset = PredecirFarmacia.objects.filter(
        fecha_nacimiento__isnull=False,
        sexo__isnull=False,
        coste_farmaceutico__isnull=False,
        coste_farmaceutico__gt=0
    )
    
    if queryset.count() < 50:
        print("❌ No hay suficientes datos para farmacia")
        return None
    
    print(f"📊 Registros encontrados: {queryset.count()}")
    
    # Preparar datos
    data = []
    for paciente in queryset:
        if paciente.edad is not None:
            try:
                data.append({
                    'edad': float(paciente.edad),
                    'sexo_num': 1 if paciente.sexo in ['M', 'H'] else 0,
                    'anyos_con_dm2': float(paciente.anyos_con_dm2 or 0),
                    'num_comorbilidades': float(paciente.num_comorbilidades or 0),
                    'num_complicaciones': float(paciente.num_complicaciones or 0),
                    'gravedad_1': float(paciente.gravedad_1 or 0),
                    'gravedad_2': float(paciente.gravedad_2 or 0),
                    'gravedad_3': float(paciente.gravedad_3 or 0),
                    'hta': float(paciente.hta or 0),
                    'dislipemias': float(paciente.dislipemias or 0),
                    'obesidad': float(paciente.obesidad or 0),
                    'ic': float(paciente.ic or 0),
                    'erc': float(paciente.erc or 0),
                    'coste_farmaceutico': float(paciente.coste_farmaceutico)
                })
            except (ValueError, TypeError) as e:
                print(f"⚠️ Saltando registro con error de conversión: {e}")
                continue
    
    df = pd.DataFrame(data)
    
    if len(df) < 50:
        print(f"❌ Datos insuficientes después de limpieza: {len(df)} registros")
        return None
        
    print(f"📊 Datos válidos: {len(df)} registros")
    features = ['edad', 'sexo_num', 'anyos_con_dm2', 'num_comorbilidades', 'num_complicaciones', 
               'gravedad_1', 'gravedad_2', 'gravedad_3', 'hta', 'dislipemias', 'obesidad', 'ic', 'erc']
    
    # Entrenar modelo
    X = df[features]
    y = df['coste_farmaceutico']
    
    # Verificar que no hay valores NaN o infinitos
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    y = y.fillna(y.median()).replace([np.inf, -np.inf], y.median())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    
    # Generar tabla
    return generar_tabla_coeficientes(modelo, X_train, y_train, y_test, y_pred, features, "Gasto Farmacéutico")

def preparar_y_analizar_coste_total():
    """Preparar datos y generar tabla para predicción coste total"""
    print("\n" + "="*60)
    print("GENERANDO TABLA: PREDICCIÓN COSTE TOTAL")
    print("="*60)
    
    # Cargar datos
    queryset = PredecirCosteTotal.objects.filter(
        fecha_nacimiento__isnull=False,
        sexo__isnull=False,
        coste_total_paciente__isnull=False,
        coste_total_paciente__gt=0
    )
    
    if queryset.count() < 50:
        print("❌ No hay suficientes datos para coste total")
        return None
    
    print(f"📊 Registros encontrados: {queryset.count()}")
    
    # Preparar datos
    data = []
    for paciente in queryset:
        if paciente.edad is not None:
            try:
                data.append({
                    'edad': float(paciente.edad),
                    'sexo_num': 1 if paciente.sexo in ['M', 'H'] else 0,
                    'anyos_con_dm2': float(paciente.anyos_con_dm2 or 0),
                    'num_comorbilidades': float(paciente.num_comorbilidades or 0),
                    'num_complicaciones': float(paciente.num_complicaciones or 0),
                    'gravedad_1': float(paciente.gravedad_1 or 0),
                    'gravedad_2': float(paciente.gravedad_2 or 0),
                    'gravedad_3': float(paciente.gravedad_3 or 0),
                    'hta': float(paciente.hta or 0),
                    'dislipemias': float(paciente.dislipemias or 0),
                    'ic': float(paciente.ic or 0),
                    'erc': float(paciente.erc or 0),
                    'coste_total_paciente': float(paciente.coste_total_paciente)
                })
            except (ValueError, TypeError) as e:
                print(f"⚠️ Saltando registro con error de conversión: {e}")
                continue
    
    df = pd.DataFrame(data)
    
    if len(df) < 50:
        print(f"❌ Datos insuficientes después de limpieza: {len(df)} registros")
        return None
        
    print(f"📊 Datos válidos: {len(df)} registros")
    features = ['edad', 'sexo_num', 'anyos_con_dm2', 'num_comorbilidades', 'num_complicaciones',
               'gravedad_1', 'gravedad_2', 'gravedad_3', 'hta', 'dislipemias', 'ic', 'erc']
    
    # Entrenar modelo
    X = df[features]
    y = df['coste_total_paciente']
    
    # Verificar que no hay valores NaN o infinitos
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    y = y.fillna(y.median()).replace([np.inf, -np.inf], y.median())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    
    # Generar tabla
    return generar_tabla_coeficientes(modelo, X_train, y_train, y_test, y_pred, features, "Coste Total")

def main():
    """Función principal - genera solo las dos tablas de coeficientes"""
    print("🚀 GENERADOR DE TABLAS DE COEFICIENTES - MODELOS DM2 (VERSIÓN CORREGIDA)")
    print(f"📁 Directorio: {os.getcwd()}")
    print("🎯 Objetivo: Generar 2 tablas de coeficientes y p-valores")
    print("🔧 Correcciones: Cálculo mejorado de p-valores y manejo de matrices singulares")
    
    archivos_generados = []
    
    try:
        # Generar tabla 1: Gasto Farmacéutico
        archivo1 = preparar_y_analizar_farmacia()
        if archivo1:
            archivos_generados.append(archivo1)
        
        # Generar tabla 2: Coste Total
        archivo2 = preparar_y_analizar_coste_total()
        if archivo2:
            archivos_generados.append(archivo2)
        
        # Resumen final
        print("\n" + "="*80)
        print("📋 RESUMEN FINAL")
        print("="*80)
        print(f"✅ Tablas generadas: {len(archivos_generados)}/2")
        
        for archivo in archivos_generados:
            print(f"   📊 {archivo}")
        
        if len(archivos_generados) == 2:
            print("\n🎉 ¡PROCESO COMPLETADO EXITOSAMENTE!")
            print("💡 Las tablas muestran coeficientes ordenados por impacto")
            print("📈 Colores indican nivel de significancia estadística")
        else:
            print("\n⚠️  Proceso incompleto - revisa los datos disponibles")
    
    except Exception as e:
        print(f"❌ ERROR GENERAL: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
