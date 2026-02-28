from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

def handle_imbalance(X, y):
    """Applies SMOTE to balance abnormal NRW vs normal conditions[cite: 562]."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)
    
    return X_res, y_res, scaler
