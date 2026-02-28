from sklearn.metrics import classification_report, precision_recall_curve, auc

def evaluate_detection(model, X_test, y_test):
    """Evaluates performance using metrics sensitive to minority classes[cite: 562, 1137]."""
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Calculate PR-AUC [cite: 371, 373]
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall, precision)
    
    print(f"PR-AUC: {pr_auc}")
    print(classification_report(y_test, y_pred))
