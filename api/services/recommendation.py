def get_clinical_recommendation(disease: str):
    """
    Maps disease prediction to Clinical Risk and Recommendation.
    """
    if disease in ["MEL", "BCC", "AKIEC"]:
        risk = "High"
        rec = "Immediate dermatologist consultation recommended for biopsy."
    elif disease in ["BKL", "DF"]:
        risk = "Medium"
        rec = "Professional evaluation recommended."
    else:
        risk = "Low"
        rec = "Routine monitoring advised."
        
    return risk, rec
