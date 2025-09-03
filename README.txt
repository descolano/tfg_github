# TFG: Análisis Epidemiológico de Diabetes Mellitus Tipo 2
Sistema de visualización y análisis de datos sanitarios desarrollado en Django para gestores del ámbito sanitario público.

## Estructura del Código

### Backend (Lógica de Negocio)
- **`./dashboard/views.py`** - Contiene la lógica de las visualizaciones, análisis estadísticos y servicios web de predicción
- **`./dashboard/models.py`** - Define los modelos de datos epidemiológicos y estructura de la base de datos utilizando el ORM de Django
- **`./dashboard/urls.py`** - Configuración de rutas y endpoints del sistema
- **`./miweb/settings.py`** - Configuración general del proyecto Django

### Frontend (Interfaces de Usuario)
- **`./dashboard/templates/`** - Contiene las plantillas HTML del cuadro de mando y todas las interfaces web interactivas

### Documentación Técnica
- **`./tabla_anexo_C/anexo_C.svg`** - Diagrama completo de la base de datos con todas las variables y relaciones
- **`./SQL_anexo_D/`** - Scripts SQL para la creación y configuración de las tablas en PostgreSQL

## Tecnologías Utilizadas
- **Backend:** Python 3.x, Django (patrón MVT)
- **Base de Datos:** PostgreSQL
- **Frontend:** HTML, CSS, JavaScript
- **Visualización:** Librerías de gráficos interactivos

## Notas para Evaluación
Este repositorio contiene el código completo del sistema desarrollado como parte del Trabajo de Fin de Grado. Los archivos principales para la evaluación técnica son los mencionados anteriormente, especialmente `views.py` y `models.py` que concentran la mayor parte de la lógica implementada.