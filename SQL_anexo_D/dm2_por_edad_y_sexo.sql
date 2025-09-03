CREATE TABLE dm2_por_edad_y_sexo AS
WITH poblacion_total AS (
    -- Calcular población total por rangos de edad y sexo
    SELECT 
        CASE 
            WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, fecha_nacimiento)) BETWEEN 18 AND 29 THEN '18-29'
            WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, fecha_nacimiento)) BETWEEN 30 AND 39 THEN '30-39'
            WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, fecha_nacimiento)) BETWEEN 40 AND 49 THEN '40-49'
            WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, fecha_nacimiento)) BETWEEN 50 AND 59 THEN '50-59'
            WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, fecha_nacimiento)) BETWEEN 60 AND 69 THEN '60-69'
            WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, fecha_nacimiento)) BETWEEN 70 AND 79 THEN '70-79'
            WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, fecha_nacimiento)) BETWEEN 80 AND 89 THEN '80-89'
            WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, fecha_nacimiento)) >= 90 THEN '90>='
        END AS rango_edad,
        sexo,
        COUNT(*) as total_poblacion
    FROM public.poblac_adulta_202401
    WHERE EXTRACT(YEAR FROM AGE(CURRENT_DATE, fecha_nacimiento)) >= 18
    GROUP BY 
        CASE 
            WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, fecha_nacimiento)) BETWEEN 18 AND 29 THEN '18-29'
            WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, fecha_nacimiento)) BETWEEN 30 AND 39 THEN '30-39'
            WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, fecha_nacimiento)) BETWEEN 40 AND 49 THEN '40-49'
            WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, fecha_nacimiento)) BETWEEN 50 AND 59 THEN '50-59'
            WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, fecha_nacimiento)) BETWEEN 60 AND 69 THEN '60-69'
            WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, fecha_nacimiento)) BETWEEN 70 AND 79 THEN '70-79'
            WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, fecha_nacimiento)) BETWEEN 80 AND 89 THEN '80-89'
            WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, fecha_nacimiento)) >= 90 THEN '90>='
        END,
        sexo
),
casos_dm2 AS (
    -- Calcular casos de DM2 por rangos de edad y sexo
    SELECT 
        CASE 
            WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, dm2.fecha_nacimiento)) BETWEEN 18 AND 29 THEN '18-29'
            WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, dm2.fecha_nacimiento)) BETWEEN 30 AND 39 THEN '30-39'
            WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, dm2.fecha_nacimiento)) BETWEEN 40 AND 49 THEN '40-49'
            WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, dm2.fecha_nacimiento)) BETWEEN 50 AND 59 THEN '50-59'
            WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, dm2.fecha_nacimiento)) BETWEEN 60 AND 69 THEN '60-69'
            WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, dm2.fecha_nacimiento)) BETWEEN 70 AND 79 THEN '70-79'
            WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, dm2.fecha_nacimiento)) BETWEEN 80 AND 89 THEN '80-89'
            WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, dm2.fecha_nacimiento)) >= 90 THEN '90>='
        END AS rango_edad,
        pob.sexo,
        COUNT(*) as casos_dm2
    FROM public.poblac_dm2_2024 dm2
    INNER JOIN public.poblac_adulta_202401 pob ON dm2.numsipcod = pob.numsipcod
    WHERE EXTRACT(YEAR FROM AGE(CURRENT_DATE, dm2.fecha_nacimiento)) >= 18
    GROUP BY 
        CASE 
            WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, dm2.fecha_nacimiento)) BETWEEN 18 AND 29 THEN '18-29'
            WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, dm2.fecha_nacimiento)) BETWEEN 30 AND 39 THEN '30-39'
            WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, dm2.fecha_nacimiento)) BETWEEN 40 AND 49 THEN '40-49'
            WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, dm2.fecha_nacimiento)) BETWEEN 50 AND 59 THEN '50-59'
            WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, dm2.fecha_nacimiento)) BETWEEN 60 AND 69 THEN '60-69'
            WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, dm2.fecha_nacimiento)) BETWEEN 70 AND 79 THEN '70-79'
            WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, dm2.fecha_nacimiento)) BETWEEN 80 AND 89 THEN '80-89'
            WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, dm2.fecha_nacimiento)) >= 90 THEN '90>='
        END,
        pob.sexo
)
SELECT 
    "edad",
    "pacientes_hombre",
    "prevalencia_hombre",
    "pacientes_mujer",
    "prevalencia_mujer"
FROM (
    -- Datos por rangos de edad
    SELECT 
        pt.rango_edad AS "edad",
        COALESCE(dm2_h.casos_dm2, 0) AS "pacientes_hombre",
        ROUND(
            CASE 
                WHEN pt_h.total_poblacion > 0 THEN 
                    (COALESCE(dm2_h.casos_dm2, 0)::numeric / pt_h.total_poblacion * 100)
                ELSE 0 
            END, 2
        ) || '%' AS "prevalencia_hombre",
        COALESCE(dm2_m.casos_dm2, 0) AS "pacientes_mujer",
        ROUND(
            CASE 
                WHEN pt_m.total_poblacion > 0 THEN 
                    (COALESCE(dm2_m.casos_dm2, 0)::numeric / pt_m.total_poblacion * 100)
                ELSE 0 
            END, 2
        ) || '%' AS "prevalencia_mujer",
        CASE pt.rango_edad
            WHEN '18-29' THEN 1
            WHEN '30-39' THEN 2
            WHEN '40-49' THEN 3
            WHEN '50-59' THEN 4
            WHEN '60-69' THEN 5
            WHEN '70-79' THEN 6
            WHEN '80-89' THEN 7
            WHEN '90>=' THEN 8
            ELSE 9
        END as orden
    FROM (
        SELECT DISTINCT rango_edad 
        FROM poblacion_total 
        WHERE rango_edad IS NOT NULL
    ) pt
    LEFT JOIN poblacion_total pt_h ON pt.rango_edad = pt_h.rango_edad AND pt_h.sexo = 'H'
    LEFT JOIN poblacion_total pt_m ON pt.rango_edad = pt_m.rango_edad AND pt_m.sexo = 'M'
    LEFT JOIN casos_dm2 dm2_h ON pt.rango_edad = dm2_h.rango_edad AND dm2_h.sexo = 'H'
    LEFT JOIN casos_dm2 dm2_m ON pt.rango_edad = dm2_m.rango_edad AND dm2_m.sexo = 'M'

    UNION ALL

    -- Fila de totales
    SELECT 
        'Total' AS "Edad (rangos)",
        SUM(COALESCE(dm2_h.casos_dm2, 0)) AS "Hombre DM2 Número",
        ROUND(
            CASE 
                WHEN SUM(pt_h.total_poblacion) > 0 THEN 
                    (SUM(COALESCE(dm2_h.casos_dm2, 0))::numeric / SUM(pt_h.total_poblacion) * 100)
                ELSE 0 
            END, 2
        ) || '%' AS "Hombre DM2 Prevalencia",
        SUM(COALESCE(dm2_m.casos_dm2, 0)) AS "Mujer DM2 Número",
        ROUND(
            CASE 
                WHEN SUM(pt_m.total_poblacion) > 0 THEN 
                    (SUM(COALESCE(dm2_m.casos_dm2, 0))::numeric / SUM(pt_m.total_poblacion) * 100)
                ELSE 0 
            END, 2
        ) || '%' AS "Mujer DM2 Prevalencia",
        9 as orden
    FROM (
        SELECT DISTINCT rango_edad 
        FROM poblacion_total 
        WHERE rango_edad IS NOT NULL
    ) pt
    LEFT JOIN poblacion_total pt_h ON pt.rango_edad = pt_h.rango_edad AND pt_h.sexo = 'H'
    LEFT JOIN poblacion_total pt_m ON pt.rango_edad = pt_m.rango_edad AND pt_m.sexo = 'M'
    LEFT JOIN casos_dm2 dm2_h ON pt.rango_edad = dm2_h.rango_edad AND dm2_h.sexo = 'H'
    LEFT JOIN casos_dm2 dm2_m ON pt.rango_edad = dm2_m.rango_edad AND dm2_m.sexo = 'M'
) datos_ordenados
