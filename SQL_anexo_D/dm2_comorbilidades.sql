CREATE TABLE dm2_comorbilidades AS
SELECT comorbilidad, pacientes, prevalencia
FROM (
    SELECT 'hta' AS comorbilidad, COUNT(*) FILTER (WHERE hta = 1) AS pacientes, 
           ROUND(100.0 * COUNT(*) FILTER (WHERE hta = 1) / COUNT(*), 2) AS prevalencia 
    FROM dm2_enfermedades WHERE dm2 = 1
    UNION ALL
    SELECT 'dislipemias', COUNT(*) FILTER (WHERE dislipemias = 1), 
           ROUND(100.0 * COUNT(*) FILTER (WHERE dislipemias = 1) / COUNT(*), 2) 
    FROM dm2_enfermedades WHERE dm2 = 1
    UNION ALL
    SELECT 'obesidad', COUNT(*) FILTER (WHERE obesidad = 1), 
           ROUND(100.0 * COUNT(*) FILTER (WHERE obesidad = 1) / COUNT(*), 2) 
    FROM dm2_enfermedades WHERE dm2 = 1
    UNION ALL
    SELECT 'ansiedad', COUNT(*) FILTER (WHERE ansiedad = 1), 
           ROUND(100.0 * COUNT(*) FILTER (WHERE ansiedad = 1) / COUNT(*), 2) 
    FROM dm2_enfermedades WHERE dm2 = 1
    UNION ALL
    SELECT 'insomnio', COUNT(*) FILTER (WHERE insomnio = 1), 
           ROUND(100.0 * COUNT(*) FILTER (WHERE insomnio = 1) / COUNT(*), 2) 
    FROM dm2_enfermedades WHERE dm2 = 1
    UNION ALL
    SELECT 'depresion', COUNT(*) FILTER (WHERE depresion = 1), 
           ROUND(100.0 * COUNT(*) FILTER (WHERE depresion = 1) / COUNT(*), 2) 
    FROM dm2_enfermedades WHERE dm2 = 1
    UNION ALL
    SELECT 'demencia', COUNT(*) FILTER (WHERE demencia = 1), 
           ROUND(100.0 * COUNT(*) FILTER (WHERE demencia = 1) / COUNT(*), 2) 
    FROM dm2_enfermedades WHERE dm2 = 1
    UNION ALL
    SELECT 'alzheimer', COUNT(*) FILTER (WHERE alzheimer = 1), 
           ROUND(100.0 * COUNT(*) FILTER (WHERE alzheimer = 1) / COUNT(*), 2) 
    FROM dm2_enfermedades WHERE dm2 = 1
    UNION ALL
    SELECT 'ic', COUNT(*) FILTER (WHERE ic = 1), 
           ROUND(100.0 * COUNT(*) FILTER (WHERE ic = 1) / COUNT(*), 2) 
    FROM dm2_enfermedades WHERE dm2 = 1
    UNION ALL
    SELECT 'fa', COUNT(*) FILTER (WHERE fa = 1), 
           ROUND(100.0 * COUNT(*) FILTER (WHERE fa = 1) / COUNT(*), 2) 
    FROM dm2_enfermedades WHERE dm2 = 1
    UNION ALL
    SELECT 'artrosis', COUNT(*) FILTER (WHERE artrosis = 1), 
           ROUND(100.0 * COUNT(*) FILTER (WHERE artrosis = 1) / COUNT(*), 2) 
    FROM dm2_enfermedades WHERE dm2 = 1
    UNION ALL
    SELECT 'epoc', COUNT(*) FILTER (WHERE epoc = 1), 
           ROUND(100.0 * COUNT(*) FILTER (WHERE epoc = 1) / COUNT(*), 2) 
    FROM dm2_enfermedades WHERE dm2 = 1
    UNION ALL
    SELECT 'asma', COUNT(*) FILTER (WHERE asma_epoc = 1), 
           ROUND(100.0 * COUNT(*) FILTER (WHERE asma_epoc = 1) / COUNT(*), 2) 
    FROM dm2_enfermedades WHERE dm2 = 1
    UNION ALL
    SELECT 'osteoporosis', COUNT(*) FILTER (WHERE osteoporosis = 1), 
           ROUND(100.0 * COUNT(*) FILTER (WHERE osteoporosis = 1) / COUNT(*), 2) 
    FROM dm2_enfermedades WHERE dm2 = 1
) sub
ORDER BY prevalencia DESC;
