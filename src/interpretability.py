import shap

def explain_model(model, X_df):
    """Generates SHAP values to improve transparency and managerial trust[cite: 563, 1211]."""
    # Using the XGBoost component for tree-based SHAP analysis [cite: 377]
    xgb_model = model.named_estimators_['xgb']
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_df)
    
    # Global feature importance plot [cite: 381, 384]
    shap.summary_plot(shap_values, X_df)
