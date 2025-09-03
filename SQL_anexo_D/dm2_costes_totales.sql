CREATE TABLE dm2_costes_totales AS
SELECT 
    -- Coste total CATP (Atención Primaria)
    SUM(
        COALESCE(catp_mfc_cen_coste, 0) +
        COALESCE(catp_mfc_domi_coste, 0) +
        COALESCE(catp_mfc_tel_coste, 0) +
        COALESCE(catp_enf_cen_coste, 0) +
        COALESCE(catp_enf_domi_coste, 0) +
        COALESCE(catp_enf_tel_coste, 0) +
        COALESCE(catp_atc_cen_coste, 0) +
        COALESCE(catp_atc_domi_coste, 0) +
        COALESCE(catp_atc_tel_coste, 0) +
        COALESCE(catp_u_apoyo_coste, 0)
    ) AS "CT_CATP",
    
    -- Coste total CEX (Consultas Externas)
    SUM(COALESCE(cex_total_coste, 0)) AS "CT_CEX",
    
    -- Coste total Urgencias
    SUM(COALESCE(urgencias_coste, 0)) AS "CT_urgencias",
    
    -- Coste total Ingresos
    SUM(COALESCE(ing_total_coste, 0)) AS "CT_ingresos",
    
    -- Coste total Importe Farmacéutico
    SUM(COALESCE(importe, 0)) AS "CT_importe_farmacéutico",
    
    -- Coste Total General
    SUM(
        COALESCE(catp_mfc_cen_coste, 0) +
        COALESCE(catp_mfc_domi_coste, 0) +
        COALESCE(catp_mfc_tel_coste, 0) +
        COALESCE(catp_enf_cen_coste, 0) +
        COALESCE(catp_enf_domi_coste, 0) +
        COALESCE(catp_enf_tel_coste, 0) +
        COALESCE(catp_atc_cen_coste, 0) +
        COALESCE(catp_atc_domi_coste, 0) +
        COALESCE(catp_atc_tel_coste, 0) +
        COALESCE(catp_u_apoyo_coste, 0) +
        COALESCE(cex_total_coste, 0) +
        COALESCE(urgencias_coste, 0) +
        COALESCE(ing_total_coste, 0) +
        COALESCE(importe, 0)
    ) AS "coste_total"
    
FROM dm2_actv_costes;
