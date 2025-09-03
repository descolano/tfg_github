CREATE TABLE pacientes_dm2_completos AS
SELECT 
    pa.numsipcod,
    pa.fecha_nacimiento,
    pa.sexo,
    
    -- AÑOS CON DIABETES TIPO 2 (desde fecha_dm2 hasta 31/12/2024)
    CASE 
        WHEN p24.fecha_dm2 IS NOT NULL THEN 
            EXTRACT(YEAR FROM AGE('2024-12-31'::date, p24.fecha_dm2))
        ELSE NULL 
    END as anyos_con_dm2,
    
    p24.anyos_dm2_rango,
    c.ing_total_num,
    e.epoc,
    e.asma_epoc,
    e.anemias,
    e.tabaco,
    e.osteoporosis,
    e.insuficiencia_venosa,
    e.hta,
    e.dislipemias,
    e.obesidad,
    e.erc,
    e.ic,
    e.ansiedad,
    e.insomnio,
    e.depresion,
    e.demencia,
    e.alzheimer,
    e.fa,
    e.artrosis,
    
    -- NÚMERO TOTAL DE COMORBILIDADES
    (COALESCE(e.epoc, 0) + COALESCE(e.asma_epoc, 0) + COALESCE(e.anemias, 0) + 
     COALESCE(e.tabaco, 0) + COALESCE(e.osteoporosis, 0) + COALESCE(e.insuficiencia_venosa, 0) + 
     COALESCE(e.hta, 0) + COALESCE(e.dislipemias, 0) + COALESCE(e.obesidad, 0) + 
     COALESCE(e.erc, 0) + COALESCE(e.ic, 0) + COALESCE(e.ansiedad, 0) + 
     COALESCE(e.insomnio, 0) + COALESCE(e.depresion, 0) + COALESCE(e.demencia, 0) + 
     COALESCE(e.alzheimer, 0) + COALESCE(e.fa, 0) + COALESCE(e.artrosis, 0)) as num_comorbilidades,
    
    -- COMPLICACIONES (desde dm2_enfermedades)
    e.cardiopatia_isquemica,
    e.enf_arterial_periferica,
    e.enf_cerebrovascular,
    e.nefropatia_diabetica,
    e.retinopatia_diabetica,
    e.neuropatia_diabetica,
    e.c_microvasculares,
    e.c_macrovasculares,
    
    -- NÚMERO TOTAL DE COMPLICACIONES
    (COALESCE(e.cardiopatia_isquemica, 0) + COALESCE(e.enf_arterial_periferica, 0) + 
     COALESCE(e.enf_cerebrovascular, 0) + COALESCE(e.nefropatia_diabetica, 0) + 
     COALESCE(e.retinopatia_diabetica, 0) + COALESCE(e.neuropatia_diabetica, 0) + 
     COALESCE(e.c_microvasculares, 0) + COALESCE(e.c_macrovasculares, 0)) as num_complicaciones,
    
    -- COSTES DESGLOSADOS POR CONCEPTO
    COALESCE(c.catp_atc_cen_coste, 0) as coste_catp_cen,
    COALESCE(c.catp_atc_domi_coste, 0) as coste_catp_domi,
    COALESCE(c.catp_atc_tel_coste, 0) as coste_catp_tel,
    COALESCE(c.catp_u_apoyo_coste, 0) as coste_catp_apoyo,
    COALESCE(c.cex_total_coste, 0) as coste_cex,
    COALESCE(c.urgencias_coste, 0) as coste_urgencias,
    COALESCE(c.ing_total_coste, 0) as coste_ingresos,
    COALESCE(c.importe, 0) as coste_farmaceutico,
    
    -- COSTE TOTAL CALCULADO
    (COALESCE(c.catp_atc_cen_coste, 0) + 
     COALESCE(c.catp_atc_domi_coste, 0) + 
     COALESCE(c.catp_atc_tel_coste, 0) + 
     COALESCE(c.catp_u_apoyo_coste, 0) + 
     COALESCE(c.cex_total_coste, 0) + 
     COALESCE(c.urgencias_coste, 0) + 
     COALESCE(c.ing_total_coste, 0) + 
     COALESCE(c.importe, 0)) as coste_total_paciente
FROM poblac_adulta_202401 pa
INNER JOIN dm2_enfermedades e ON pa.numsipcod = e.numsipcod
LEFT JOIN poblac_dm2_2024 p24 ON pa.numsipcod = p24.numsipcod
LEFT JOIN dm2_actv_costes c ON pa.numsipcod = c.numsipcod
