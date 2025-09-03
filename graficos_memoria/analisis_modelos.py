#!/usr/bin/env python3
"""
Script completo de análisis de modelos DM2 - 4 análisis separados
Ejecutar desde la raíz del proyecto Django
"""

import os
import sys
import django
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configurar Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'miweb.settings')
django.setup()

from dashboard.models import PredecirFarmacia, PredecirCosteTotal

def calcular_mape(y_true, y_pred):
    """Calcular Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calcular_coeficientes_pvalores(modelo, X):
    """Calcular coeficientes y p-valores para regresión lineal"""
    # Calcular estadísticas t
    mse = np.mean((y_test - modelo.predict(X_test)) ** 2)
    var_beta = mse * (np.linalg.inv(X.T.dot(X)).diagonal())
    sd_beta = np.sqrt(var_beta)
    ts_beta = modelo.coef_ / sd_beta
    p_values = [2 * (1 - stats.t.cdf(np.abs(t), len(X) - len(modelo.coef_) - 1)) for t in ts_beta]
    
    return {
        'coeficientes': modelo.coef_,
        'p_valores': p_values,
        'intercepto': modelo.intercept_
    }

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

def analizar_regresion_lineal(df, features, target_col, titulo, aplicar_log=True):
    """Análisis detallado de regresión lineal con opción logarítmica"""
    print(f"\n{'='*60}")
    print(f"ANÁLISIS REGRESIÓN LINEAL - {titulo}")
    print(f"Transformación logarítmica: {'SÍ' if aplicar_log else 'NO'}")
    print(f"{'='*60}")
    
    X = df[features]
    y = df[target_col]
    
    if aplicar_log:
        y_transformed = np.log1p(y)
        transform_label = "Log-transformada"
    else:
        y_transformed = y
        transform_label = "Original"
    
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.2, random_state=42)
    
    # Entrenar modelo
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    
    # Predicciones
    y_pred_transformed = modelo.predict(X_test)
    
    # Convertir predicciones a escala original si se aplicó log
    if aplicar_log:
        y_pred = np.expm1(y_pred_transformed)
        y_test_original = np.expm1(y_test)
    else:
        y_pred = y_pred_transformed
        y_test_original = y_test
    
    # Calcular métricas en escala original
    r2 = r2_score(y_test_original, y_pred)
    mse = mean_squared_error(y_test_original, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, y_pred)
    mape = calcular_mape(y_test_original, y_pred)
    
    # Métricas adicionales
    residuos = y_test_original - y_pred
    residuos_std = residuos / np.std(residuos)
    
    # Calcular coeficientes y p-valores
    try:
        n = len(X_test)
        p = X_test.shape[1]
        mse_reg = np.sum((y_test - y_pred_transformed) ** 2) / (n - p - 1)
        
        # Matriz de covarianza de los coeficientes
        X_test_with_intercept = np.column_stack([np.ones(len(X_test)), X_test])
        cov_matrix = mse_reg * np.linalg.inv(X_test_with_intercept.T @ X_test_with_intercept)
        std_errors = np.sqrt(np.diag(cov_matrix))
        
        # t-statistics y p-valores
        coefs_with_intercept = np.append(modelo.intercept_, modelo.coef_)
        t_stats = coefs_with_intercept / std_errors
        p_valores = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))
        
        coef_info = {
            'coeficientes': modelo.coef_,
            'p_valores': p_valores[1:],  # Excluir intercepto
            'intercepto': modelo.intercept_,
            'p_valor_intercepto': p_valores[0]
        }
    except:
        coef_info = None
    
    # Mostrar métricas
    print(f"\nMÉTRICAS:")
    print(f"  R²: {r2:.4f}")
    print(f"  MSE: {mse:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Desv. Estándar Residuos: {np.std(residuos):.2f}")
    
    if coef_info:
        print(f"\nCOEFICIENTES:")
        print(f"  Intercepto: {coef_info['intercepto']:.4f} (p={coef_info['p_valor_intercepto']:.4f})")
        for i, (coef, p_val, feature) in enumerate(zip(coef_info['coeficientes'], coef_info['p_valores'], features)):
            significancia = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"  {feature}: {coef:.4f} (p={p_val:.4f}) {significancia}")
    
    # Crear gráficos
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Regresión Lineal - {titulo} ({transform_label})', fontsize=16, fontweight='bold')
    
    # 1. Predicciones vs Reales
    axes[0, 0].scatter(y_test_original, y_pred, alpha=0.6, color='blue', s=30)
    axes[0, 0].plot([y_test_original.min(), y_test_original.max()], 
                    [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Valores Reales (€)')
    axes[0, 0].set_ylabel('Predicciones (€)')
    axes[0, 0].set_title(f'Predicciones vs Reales (R²={r2:.3f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Distribución de residuos
    axes[0, 1].hist(residuos, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 1].set_xlabel('Residuos (€)')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].set_title(f'Distribución de Residuos (MAE={mae:.1f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Q-Q plot de residuos
    stats.probplot(residuos_std, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot Residuos Estandarizados')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Residuos vs Predicciones
    axes[1, 1].scatter(y_pred, residuos, alpha=0.6, color='blue', s=30)
    axes[1, 1].axhline(y=0, color='red', linestyle='--')
    axes[1, 1].set_xlabel('Predicciones (€)')
    axes[1, 1].set_ylabel('Residuos (€)')
    axes[1, 1].set_title(f'Residuos vs Predicciones (MAPE={mape:.1f}%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'analisis_lineal_{titulo.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado: {filename}")
    plt.show()
    
    return {
        'R²': r2, 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape,
        'coef_info': coef_info, 'transform_applied': aplicar_log
    }

def analizar_random_forest(df, features, target_col, titulo):
    """Análisis detallado de Random Forest"""
    print(f"\n{'='*60}")
    print(f"ANÁLISIS RANDOM FOREST - {titulo}")
    print(f"{'='*60}")
    
    X = df[features]
    y = df[target_col]
    
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar modelo
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    
    # Predicciones
    y_pred = modelo.predict(X_test)
    
    # Calcular métricas
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = calcular_mape(y_test, y_pred)
    
    # Métricas adicionales
    residuos = y_test - y_pred
    residuos_std = residuos / np.std(residuos)
    
    # Importancia de características
    importances = modelo.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Mostrar métricas
    print(f"\nMÉTRICAS:")
    print(f"  R²: {r2:.4f}")
    print(f"  MSE: {mse:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Desv. Estándar Residuos: {np.std(residuos):.2f}")
    
    print(f"\nTOP 5 CARACTERÍSTICAS:")
    for i in range(min(5, len(features))):
        idx = indices[i]
        print(f"  {features[idx]}: {importances[idx]:.4f}")
    
    # Crear gráficos
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Random Forest - {titulo}', fontsize=16, fontweight='bold')
    
    # 1. Predicciones vs Reales
    axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='green', s=30)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Valores Reales (€)')
    axes[0, 0].set_ylabel('Predicciones (€)')
    axes[0, 0].set_title(f'Predicciones vs Reales (R²={r2:.3f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Distribución de residuos
    axes[0, 1].hist(residuos, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xlabel('Residuos (€)')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].set_title(f'Distribución de Residuos (MAE={mae:.1f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Importancia de características
    n_features_plot = min(10, len(features))
    axes[1, 0].bar(range(n_features_plot), importances[indices[:n_features_plot]], color='green', alpha=0.7)
    axes[1, 0].set_title('Importancia de Características')
    axes[1, 0].set_xlabel('Características')
    axes[1, 0].set_ylabel('Importancia')
    axes[1, 0].set_xticks(range(n_features_plot))
    axes[1, 0].set_xticklabels([features[i] for i in indices[:n_features_plot]], rotation=45, ha='right')
    
    # 4. Residuos vs Predicciones
    axes[1, 1].scatter(y_pred, residuos, alpha=0.6, color='green', s=30)
    axes[1, 1].axhline(y=0, color='red', linestyle='--')
    axes[1, 1].set_xlabel('Predicciones (€)')
    axes[1, 1].set_ylabel('Residuos (€)')
    axes[1, 1].set_title(f'Residuos vs Predicciones (MAPE={mape:.1f}%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'analisis_rf_{titulo.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado: {filename}")
    plt.show()
    
    return {
        'R²': r2, 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape,
        'importances': dict(zip(features, importances))
    }

def main():
    print("INICIANDO ANÁLISIS COMPLETO DE MODELOS DM2")
    print(f"Directorio: {os.getcwd()}")
    
    try:
        resultados = {}
        
        # 1. DATOS FARMACIA
        print("\nCargando datos de farmacia...")
        df_farm, features_farm, target_farm = preparar_datos_farmacia()
        
        if df_farm is not None:
            print(f"✅ Datos farmacia: {len(df_farm)} registros")
            
            # Análisis Regresión Lineal Farmacia (CON logaritmo)
            resultados['farmacia_lineal'] = analizar_regresion_lineal(
                df_farm, features_farm, target_farm, "Farmacia", aplicar_log=True
            )
            
            # Análisis Random Forest Farmacia
            resultados['farmacia_rf'] = analizar_random_forest(
                df_farm, features_farm, target_farm, "Farmacia"
            )
        else:
            print("❌ No hay suficientes datos para farmacia")
        
        # 2. DATOS COSTE TOTAL
        print("\nCargando datos de coste total...")
        df_total, features_total, target_total = preparar_datos_coste_total()
        
        if df_total is not None:
            print(f"✅ Datos coste total: {len(df_total)} registros")
            
            # Análisis Regresión Lineal Coste Total (CON logaritmo)
            resultados['coste_total_lineal'] = analizar_regresion_lineal(
                df_total, features_total, target_total, "Coste Total", aplicar_log=True
            )
            
            # Análisis Random Forest Coste Total
            resultados['coste_total_rf'] = analizar_random_forest(
                df_total, features_total, target_total, "Coste Total"
            )
        else:
            print("❌ No hay suficientes datos para coste total")
        
        # RESUMEN FINAL
        print(f"\n{'='*80}")
        print("RESUMEN COMPARATIVO")
        print(f"{'='*80}")
        
        for nombre, resultado in resultados.items():
            print(f"\n{nombre.upper().replace('_', ' ')}:")
            print(f"  R²: {resultado['R²']:.4f}")
            print(f"  RMSE: {resultado['RMSE']:.2f}")
            print(f"  MAE: {resultado['MAE']:.2f}")
            print(f"  MAPE: {resultado['MAPE']:.2f}%")
            if 'transform_applied' in resultado:
                print(f"  Transformación Log: {'SÍ' if resultado['transform_applied'] else 'NO'}")
        
        print(f"\n{'='*80}")
        print("ANÁLISIS COMPLETADO")
        print("Archivos generados:")
        archivos = ['analisis_lineal_farmacia.png', 'analisis_rf_farmacia.png',
                   'analisis_lineal_coste_total.png', 'analisis_rf_coste_total.png']
        for archivo in archivos:
            if os.path.exists(archivo):
                print(f"  ✅ {archivo}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
