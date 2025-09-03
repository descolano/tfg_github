from django.db import models
from datetime import date

class Dm2Comorbilidades(models.Model):
    comorbilidad = models.TextField(primary_key=True)
    pacientes = models.IntegerField(null=True, blank=True)
    prevalencia = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    
    class Meta:
        db_table = 'dm2_comorbilidades'
        managed = False
        
    def __str__(self):
        return self.comorbilidad

class Dm2Complicaciones(models.Model):
    complicacion = models.TextField(primary_key=True)
    pacientes = models.IntegerField(null=True, blank=True)
    prevalencia = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    
    class Meta:
        db_table = 'dm2_complicaciones'
        managed = False
        
    def __str__(self):
        return self.complicacion

class Dm2Departamentos(models.Model):
    zona = models.TextField(primary_key=True)
    pacientes = models.IntegerField(null=True, blank=True)
    prevalencia = models.TextField(null=True, blank=True) 

    class Meta:
        db_table = 'dm2_departamentos'
        managed = False
        
    def __str__(self):
        return self.zona

class Dm2EdadSexo(models.Model):
    edad = models.TextField(primary_key=True, db_column='edad')
    pacientes_hombre = models.IntegerField(null=True, blank=True, db_column='pacientes_hombre')
    prevalencia_hombre = models.TextField(null=True, blank=True, db_column='prevalencia_hombre')
    pacientes_mujer = models.IntegerField(null=True, blank=True, db_column='pacientes_mujer')
    prevalencia_mujer = models.TextField(null=True, blank=True, db_column='prevalencia_mujer')

    class Meta:
        db_table = 'dm2_por_edad_y_sexo'
        managed = False
        
    def __str__(self):
        return self.edad

class PredecirFarmacia(models.Model):
    numsipcod = models.CharField(max_length=50, primary_key=True)
    fecha_nacimiento = models.DateField(null=True, blank=True)
    sexo = models.CharField(max_length=1, null=True, blank=True)
    
    anyos_con_dm2 = models.IntegerField(null=True, blank=True)
    num_comorbilidades = models.IntegerField(null=True, blank=True, default=0)
    num_complicaciones = models.IntegerField(null=True, blank=True, default=0)
    
    gravedad_1 = models.IntegerField(null=True, blank=True, default=0, help_text="1 si tiene exactamente 1 ingreso")
    gravedad_2 = models.IntegerField(null=True, blank=True, default=0, help_text="1 si tiene exactamente 2 ingresos")
    gravedad_3 = models.IntegerField(null=True, blank=True, default=0, help_text="1 si tiene 3 o más ingresos")
    
    epoc = models.IntegerField(null=True, blank=True, default=0)
    asma_epoc = models.IntegerField(null=True, blank=True, default=0)
    anemias = models.IntegerField(null=True, blank=True, default=0)
    tabaco = models.IntegerField(null=True, blank=True, default=0)
    osteoporosis = models.IntegerField(null=True, blank=True, default=0)
    insuficiencia_venosa = models.IntegerField(null=True, blank=True, default=0)
    hta = models.IntegerField(null=True, blank=True, default=0)
    dislipemias = models.IntegerField(null=True, blank=True, default=0)
    obesidad = models.IntegerField(null=True, blank=True, default=0)
    erc = models.IntegerField(null=True, blank=True, default=0)
    ic = models.IntegerField(null=True, blank=True, default=0)
    ansiedad = models.IntegerField(null=True, blank=True, default=0)
    insomnio = models.IntegerField(null=True, blank=True, default=0)
    depresion = models.IntegerField(null=True, blank=True, default=0)
    demencia = models.IntegerField(null=True, blank=True, default=0)
    alzheimer = models.IntegerField(null=True, blank=True, default=0)
    fa = models.IntegerField(null=True, blank=True, default=0)
    artrosis = models.IntegerField(null=True, blank=True, default=0)
    
    cardiopatia_isquemica = models.IntegerField(null=True, blank=True, default=0)
    enf_arterial_periferica = models.IntegerField(null=True, blank=True, default=0)
    enf_cerebrovascular = models.IntegerField(null=True, blank=True, default=0)
    nefropatia_diabetica = models.IntegerField(null=True, blank=True, default=0)
    retinopatia_diabetica = models.IntegerField(null=True, blank=True, default=0)
    neuropatia_diabetica = models.IntegerField(null=True, blank=True, default=0)
    c_microvasculares = models.IntegerField(null=True, blank=True, default=0)
    c_macrovasculares = models.IntegerField(null=True, blank=True, default=0)
    
    coste_farmaceutico = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True)
    
    @property
    def edad(self):
        if self.fecha_nacimiento:
            today = date.today()
            return today.year - self.fecha_nacimiento.year - ((today.month, today.day) < (self.fecha_nacimiento.month, self.fecha_nacimiento.day))
        return None
    
    @property
    def nivel_gravedad_ingresos(self):
        if self.gravedad_1:
            return 1
        elif self.gravedad_2:
            return 2
        elif self.gravedad_3:
            return 3
        return 0
    
    class Meta:
        db_table = 'pacientes_dm2_completos'
        managed = False
        
    def __str__(self):
        return f"Predictor Farmacia: {self.numsipcod}"

class PredecirCosteTotal(models.Model):
    """Modelo para predicción de coste total"""
    numsipcod = models.CharField(max_length=50, primary_key=True)
    fecha_nacimiento = models.DateField(null=True, blank=True)
    sexo = models.CharField(max_length=1, null=True, blank=True)
    
    anyos_con_dm2 = models.IntegerField(null=True, blank=True)
    num_comorbilidades = models.IntegerField(null=True, blank=True, default=0)
    num_complicaciones = models.IntegerField(null=True, blank=True, default=0)
    
    gravedad_1 = models.IntegerField(null=True, blank=True, default=0, help_text="1 si tiene exactamente 1 ingreso")
    gravedad_2 = models.IntegerField(null=True, blank=True, default=0, help_text="1 si tiene exactamente 2 ingresos")
    gravedad_3 = models.IntegerField(null=True, blank=True, default=0, help_text="1 si tiene 3 o más ingresos")
    epoc = models.IntegerField(null=True, blank=True, default=0)
    asma_epoc = models.IntegerField(null=True, blank=True, default=0)
    anemias = models.IntegerField(null=True, blank=True, default=0)
    tabaco = models.IntegerField(null=True, blank=True, default=0)
    osteoporosis = models.IntegerField(null=True, blank=True, default=0)
    insuficiencia_venosa = models.IntegerField(null=True, blank=True, default=0)
    hta = models.IntegerField(null=True, blank=True, default=0)
    dislipemias = models.IntegerField(null=True, blank=True, default=0)
    obesidad = models.IntegerField(null=True, blank=True, default=0)
    erc = models.IntegerField(null=True, blank=True, default=0)
    ic = models.IntegerField(null=True, blank=True, default=0)
    ansiedad = models.IntegerField(null=True, blank=True, default=0)
    insomnio = models.IntegerField(null=True, blank=True, default=0)
    depresion = models.IntegerField(null=True, blank=True, default=0)
    demencia = models.IntegerField(null=True, blank=True, default=0)
    alzheimer = models.IntegerField(null=True, blank=True, default=0)
    fa = models.IntegerField(null=True, blank=True, default=0)
    artrosis = models.IntegerField(null=True, blank=True, default=0)
    cardiopatia_isquemica = models.IntegerField(null=True, blank=True, default=0)
    enf_arterial_periferica = models.IntegerField(null=True, blank=True, default=0)
    enf_cerebrovascular = models.IntegerField(null=True, blank=True, default=0)
    nefropatia_diabetica = models.IntegerField(null=True, blank=True, default=0)
    retinopatia_diabetica = models.IntegerField(null=True, blank=True, default=0)
    neuropatia_diabetica = models.IntegerField(null=True, blank=True, default=0)
    c_microvasculares = models.IntegerField(null=True, blank=True, default=0)
    c_macrovasculares = models.IntegerField(null=True, blank=True, default=0)
    
    # COSTES DESGLOSADOS (para predicción de coste total)
    coste_catp_cen = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True, default=0)
    coste_catp_domi = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True, default=0)
    coste_catp_tel = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True, default=0)
    coste_catp_apoyo = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True, default=0)
    coste_cex = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True, default=0)
    coste_urgencias = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True, default=0)
    coste_ingresos = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True, default=0)
    coste_farmaceutico = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True, default=0)
    coste_total_paciente = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True)
    
    @property
    def edad(self):
        if self.fecha_nacimiento:
            today = date.today()
            return today.year - self.fecha_nacimiento.year - ((today.month, today.day) < (self.fecha_nacimiento.month, self.fecha_nacimiento.day))
        return None
    
    @property
    def nivel_gravedad_ingresos(self):
        if self.gravedad_1:
            return 1
        elif self.gravedad_2:
            return 2
        elif self.gravedad_3:
            return 3
        return 0
    
    @property
    def coste_ambulatorio(self):
        return (
            (self.coste_catp_cen or 0) + 
            (self.coste_catp_domi or 0) + 
            (self.coste_catp_tel or 0) + 
            (self.coste_catp_apoyo or 0) + 
            (self.coste_cex or 0)
        )
    
    @property
    def coste_hospitalario(self):
        return (self.coste_urgencias or 0) + (self.coste_ingresos or 0)
    
    class Meta:
        db_table = 'pacientes_dm2_completos'
        managed = False
        
    def __str__(self):
        return f"Predictor Coste Total: {self.numsipcod}"

class Dm2CostesTotales(models.Model):
    # Usar los nombres exactos de las columnas que creaste en la SQL
    ct_catp = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True, db_column='CT_CATP')
    ct_cex = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True, db_column='CT_CEX')
    ct_urgencias = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True, db_column='CT_urgencias')
    ct_ingresos = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True, db_column='CT_ingresos')
    ct_importe_farmaceutico = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True, db_column='CT_importe_farmacéutico')
    coste_total = models.DecimalField(max_digits=15, decimal_places=2, primary_key=True, db_column='coste_total')
    
    class Meta:
        db_table = 'dm2_costes_totales'
        managed = False
        verbose_name = 'Coste Total'
        verbose_name_plural = 'Costes Totales'
    
    def __str__(self):
        return f"Costes Totales: {self.coste_total}€"
