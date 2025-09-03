
CREATE TABLE dm2_complicaciones AS
SELECT complicacion, pacientes, prevalencia
FROM (
    SELECT 'ic' AS complicacion, COUNT(*) FILTER (WHERE ic = 1) AS pacientes, 
           ROUND(100.0 * COUNT(*) FILTER (WHERE ic = 1) / COUNT(*), 2) AS prevalencia 
    FROM dm2_enfermedades WHERE dm2 = 1
    UNION ALL
    SELECT 'erc', COUNT(*) FILTER (WHERE erc = 1), 
           ROUND(100.0 * COUNT(*) FILTER (WHERE erc = 1) / COUNT(*), 2) 
    FROM dm2_enfermedades WHERE dm2 = 1
    UNION ALL
    SELECT 'nefropatia_diabetica', COUNT(*) FILTER (WHERE nefropatia_diabetica = 1), 
           ROUND(100.0 * COUNT(*) FILTER (WHERE nefropatia_diabetica = 1) / COUNT(*), 2) 
    FROM dm2_enfermedades WHERE dm2 = 1
    UNION ALL
    SELECT 'retinopatia_diabetica', COUNT(*) FILTER (WHERE retinopatia_diabetica = 1), 
           ROUND(100.0 * COUNT(*) FILTER (WHERE retinopatia_diabetica = 1) / COUNT(*), 2) 
    FROM dm2_enfermedades WHERE dm2 = 1
    UNION ALL
    SELECT 'neuropatia_diabetica', COUNT(*) FILTER (WHERE neuropatia_diabetica = 1), 
           ROUND(100.0 * COUNT(*) FILTER (WHERE neuropatia_diabetica = 1) / COUNT(*), 2) 
    FROM dm2_enfermedades WHERE dm2 = 1
    UNION ALL
    SELECT 'cardiopatia_isquemica', COUNT(*) FILTER (WHERE cardiopatia_isquemica = 1), 
           ROUND(100.0 * COUNT(*) FILTER (WHERE cardiopatia_isquemica = 1) / COUNT(*), 2) 
    FROM dm2_enfermedades WHERE dm2 = 1
    UNION ALL
    SELECT 'enf_arterial_periferica', COUNT(*) FILTER (WHERE enf_arterial_periferica = 1), 
           ROUND(100.0 * COUNT(*) FILTER (WHERE enf_arterial_periferica = 1) / COUNT(*), 2) 
    FROM dm2_enfermedades WHERE dm2 = 1
    UNION ALL
    SELECT 'enf_cerebrovascular', COUNT(*) FILTER (WHERE enf_cerebrovascular = 1), 
           ROUND(100.0 * COUNT(*) FILTER (WHERE enf_cerebrovascular = 1) / COUNT(*), 2) 
    FROM dm2_enfermedades WHERE dm2 = 1
    UNION ALL
    SELECT 'insuficiencia_venosa', COUNT(*) FILTER (WHERE insuficiencia_venosa = 1), 
           ROUND(100.0 * COUNT(*) FILTER (WHERE insuficiencia_venosa = 1) / COUNT(*), 2) 
    FROM dm2_enfermedades WHERE dm2 = 1
) sub
ORDER BY prevalencia DESC;
