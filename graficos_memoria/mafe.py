#!/usr/bin/env python3
"""
Script para calcular RMSE y MAFE de los modelos DM2
Ejecutar desde la raíz del proyecto Django
"""

import os
import sys
import django
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Configurar Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'miweb.settings')
django.setup()

from dashboard.models import PredecirFarmacia, PredecirCosteTotal

def calcular_mafe(y_true, y_pred):
    """Calcular Mean Absolute Fractional Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true))

def calcular_rmse(y_true, y_pred):
    """Calcular Root Mean Square Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def preparar_datos_farmacia():
    """Cargar y preparar datos de farmacia"""
    queryset = PredecirFarmacia.objects.filter(
        fecha_nacimiento__isnull=False,
        sexo__isnull=False,
        coste_farmaceutico__isnull=False,
        coste_farmaceutico__gt=0
    )
    
    if queryset.count() < 50:
        return None, None, None
    
    data = []
    for paciente in queryset:
        if paciente.edad is not None:
            data.append({
                'edad': paciente.edad,
                'sexo_num': 1 if paciente.sexo in ['M', 'H'] else 0,
                'anyos_con_dm2': paciente.anyos_con_dm2 or 0,
                'num_comorbilidades': paciente.num_comorbilidades or 0,
                'num_complicaciones': paciente.num_complicaciones or 0,
                'gravedad_1': paciente.gravedad_1 or 0,
                'gravedad_2': paciente.gravedad_2 or 0,
                'gravedad_3': paciente.gravedad_3 or 0,
                'hta': paciente.hta or 0,
                'dislipemias': paciente.dislipemias or 0,
                'obesidad': paciente.obesidad or 0,
                'ic': paciente.ic or 0,
                'erc': paciente.erc or 0,
                'coste_farmaceutico': float(paciente.coste_farmaceutico)
            })
    
    df = pd.DataFrame(data)
    features = ['edad', 'sexo_num', 'anyos_con_dm2', 'num_comorbilidades', 'num_complicaciones', 
               'gravedad_1', 'gravedad_2', 'gravedad_3', 'hta', 'dislipemias', 'obesidad', 'ic', 'erc']
    
    return df, features, 'coste_farmaceutico'

def preparar_datos_coste_total():
    """Cargar y preparar datos de coste total"""
    queryset = PredecirCosteTotal.objects.filter(
        fecha_nacimiento__isnull=False,
        sexo__isnull=False,
        coste_total_paciente__isnull=False,
        coste_total_paciente__gt=0
    )
    
    if queryset.count() < 50:
        return None, None, None
    
    data = []
    for paciente in queryset:
        if paciente.edad is not None:
            data.append({
                'edad': paciente.edad,
                'sexo_num': 1 if paciente.sexo in ['M', 'H'] else 0,
                'anyos_con_dm2': paciente.anyos_con_dm2 or 0,
                'num_comorbilidades': paciente.num_comorbilidades or 0,
                'num_complicaciones': paciente.num_complicaciones or 0,
                'gravedad_1': paciente.gravedad_1 or 0,
                'gravedad_2': paciente.gravedad_2 or 0,
                'gravedad_3': paciente.gravedad_3 or 0,
                'hta': paciente.hta or 0,
                'dislipemias': paciente.dislipemias or 0,
                'ic': paciente.ic or 0,
                'erc': paciente.erc or 0,
                'coste_total_paciente': float(paciente.coste_total_paciente)
            })
    
    df = pd.DataFrame(data)
    features = ['edad', 'sexo_num', 'anyos_con_dm2', 'num_comorbilidades', 'num_complicaciones',
               'gravedad_1', 'gravedad_2', 'gravedad_3', 'hta', 'dislipemias', 'ic', 'erc']
    
    return df, features, 'coste_total_paciente'

def evaluar_modelo(X, y, titulo, usar_log=False):
    """Evaluar ambos modelos y calcular métricas"""
    print(f"\n{'='*60}")
    print(f"EVALUANDO: {titulo}")
    if usar_log:
        print("CON transformación logarítmica")
        y_transformed = np.log1p(y)
    else:
        print("SIN transformación logarítmica")
        y_transformed = y
    print(f"{'='*60}")
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.2, random_state=42)
    
    # REGRESIÓN LINEAL
    modelo_lineal = LinearRegression()
    modelo_lineal.fit(X_train, y_train)
    y_pred_lineal = modelo_lineal.predict(X_test)
    
    # RANDOM FOREST
    modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo_rf.fit(X_train, y_train)
    y_pred_rf = modelo_rf.predict(X_test)
    
    # Si se usó log, convertir predicciones a escala original
    if usar_log:
        y_pred_lineal = np.expm1(y_pred_lineal)
        y_pred_rf = np.expm1(y_pred_rf)
        y_test_original = np.expm1(y_test)
    else:
        y_test_original = y_test
    
    # Calcular métricas
    print("\nRESULTADOS:")
    print(f"{'MÉTRICA':<25} {'REG. LINEAL':<15} {'RANDOM FOREST':<15}")
    print("-" * 55)
    
    # R²
    r2_lineal = r2_score(y_test_original, y_pred_lineal)
    r2_rf = r2_score(y_test_original, y_pred_rf)
    print(f"{'R²':<25} {r2_lineal:<15.4f} {r2_rf:<15.4f}")
    
    # MAE
    mae_lineal = mean_absolute_error(y_test_original, y_pred_lineal)
    mae_rf = mean_absolute_error(y_test_original, y_pred_rf)
    print(f"{'MAE':<25} {mae_lineal:<15.2f} {mae_rf:<15.2f}")
    
    # RMSE
    rmse_lineal = calcular_rmse(y_test_original, y_pred_lineal)
    rmse_rf = calcular_rmse(y_test_original, y_pred_rf)
    print(f"{'RMSE':<25} {rmse_lineal:<15.2f} {rmse_rf:<15.2f}")
    
    # MAFE
    try:
        mafe_lineal = calcular_mafe(y_test_original, y_pred_lineal)
        mafe_rf = calcular_mafe(y_test_original, y_pred_rf)
        print(f"{'MAFE':<25} {mafe_lineal:<15.4f} {mafe_rf:<15.4f}")
    except:
        print(f"{'MAFE':<25} {'Error':<15} {'Error':<15}")
    
    # MAPE
    try:
        mape_lineal = np.mean(np.abs((y_test_original - y_pred_lineal) / y_test_original)) * 100
        mape_rf = np.mean(np.abs((y_test_original - y_pred_rf) / y_test_original)) * 100
        print(f"{'MAPE (%)':<25} {mape_lineal:<15.2f} {mape_rf:<15.2f}")
    except:
        print(f"{'MAPE (%)':<25} {'inf':<15} {'inf':<15}")
    
    return {
        'r2_lineal': r2_lineal, 'r2_rf': r2_rf,
        'mae_lineal': mae_lineal, 'mae_rf': mae_rf,
        'rmse_lineal': rmse_lineal, 'rmse_rf': rmse_rf,
        'mafe_lineal': mafe_lineal if 'mafe_lineal' in locals() else None,
        'mafe_rf': mafe_rf if 'mafe_rf' in locals() else None
    }

def main():
    print("CALCULANDO MÉTRICAS ADICIONALES PARA MODELOS DM2")
    print(f"Directorio: {os.getcwd()}")
    
    try:
        # FARMACIA
        print("\n" + "="*80)
        print("MODELO FARMACIA")
        print("="*80)
        
        df_farm, features_farm, target_farm = preparar_datos_farmacia()
        if df_farm is not None:
            X_farm = df_farm[features_farm]
            y_farm = df_farm[target_farm]
            
            # Sin log
            resultados_farm_sin = evaluar_modelo(X_farm, y_farm, "FARMACIA SIN LOG", usar_log=False)
            
            # Con log
            resultados_farm_con = evaluar_modelo(X_farm, y_farm, "FARMACIA CON LOG", usar_log=True)
        
        # COSTE TOTAL
        print("\n" + "="*80)
        print("MODELO COSTE TOTAL")
        print("="*80)
        
        df_total, features_total, target_total = preparar_datos_coste_total()
        if df_total is not None:
            X_total = df_total[features_total]
            y_total = df_total[target_total]
            
            # Sin log
            resultados_total_sin = evaluar_modelo(X_total, y_total, "COSTE TOTAL SIN LOG", usar_log=False)
            
            # Con log
            resultados_total_con = evaluar_modelo(X_total, y_total, "COSTE TOTAL CON LOG", usar_log=True)
        
        print("\n" + "="*80)
        print("ANÁLISIS COMPLETADO")
        print("="*80)
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
