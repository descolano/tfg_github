from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_home, name='dashboard_home'),
    path('edad-sexo/', views.prevalencia_edad_sexo, name='prevalencia_edad_sexo'),
    path('comorbilidades/', views.comorbilidades, name='comorbilidades'),
    path('prevalencia-zona/', views.prevalencia_zona, name='prevalencia_zona'),
    path('costes-totales/', views.costes_totales, name='costes_totales'),
    path('prediccion-gasto/', views.prediccion_gasto, name='prediccion_gasto'),
    path('prediccion-coste-total/', views.prediccion_coste_total, name='prediccion_coste_total'),
    path('prediccion-individual-farmacia/', views.prediccion_individual_farmacia, name='prediccion_individual_farmacia'),
    path('prediccion-individual-coste-total/', views.prediccion_individual_coste_total, name='prediccion_individual_coste_total'),
]
