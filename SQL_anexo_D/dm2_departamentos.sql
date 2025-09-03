CREATE TABLE dm2_departamentos AS
WITH poblacion_por_zona AS (
    -- Calcular población total por zona de salud
    SELECT 
        zona_desc,
        COUNT(*) as total_poblacion
    FROM public.poblac_adulta_202401
    WHERE EXTRACT(YEAR FROM AGE(CURRENT_DATE, fecha_nacimiento)) >= 18
    GROUP BY zona_desc
),
casos_dm2_por_zona AS (
    -- Calcular casos de DM2 por zona de salud
    SELECT 
        pob.zona_desc,
        COUNT(*) as casos_dm2
    FROM public.poblac_dm2_2024 dm2
    INNER JOIN public.poblac_adulta_202401 pob ON dm2.numsipcod = pob.numsipcod
    WHERE EXTRACT(YEAR FROM AGE(CURRENT_DATE, dm2.fecha_nacimiento)) >= 18
    GROUP BY pob.zona_desc
),
SELECT 
    COALESCE(pz.zona_desc, dm2z.zona_desc) AS "zona",
    COALESCE(dm2z.casos_dm2, 0) AS "pacientes",
    ROUND(
        CASE 
            WHEN pz.total_poblacion > 0 THEN 
                (COALESCE(dm2z.casos_dm2, 0)::numeric / pz.total_poblacion * 100)
            ELSE 0 
        END, 2
    ) || '%' AS "prevalencia"
FROM poblacion_por_zona pz
FULL OUTER JOIN casos_dm2_por_zona dm2z ON pz.zona_desc = dm2z.zona_desc
WHERE COALESCE(pz.zona_desc, dm2z.zona_desc) IS NOT NULL
ORDER BY 
    CASE 
        WHEN pz.total_poblacion > 0 THEN 
            (COALESCE(dm2z.casos_dm2, 0)::numeric / pz.total_poblacion * 100)
        ELSE 0 
    END DESC;
