from src import data_loader, feature_engineering, preprocessing, models, evaluation, interpretability

def run_pipeline():
    # 1. Load 24 months of DMA data [cite: 293]
    raw_data = data_loader.load_utility_data('data/raw/nairobi_dma_data.csv')
    
    # 2. IWA Water Balance & Labelling [cite: 608]
    processed_data = feature_engineering.calculate_iwa_metrics(raw_data)
    
    # 3. Features & Target
    X = processed_data.drop(columns=['target', 'nrw_volume', 'nrw_percent'])
    y = processed_data['target']
    
    # 4. Handle Imbalance (SMOTE) [cite: 141]
    X_res, y_res, scaler = preprocessing.handle_imbalance(X, y)
    
    # 5. Train Stacked Ensemble [cite: 561]
    model = models.get_stacked_ensemble()
    model.fit(X_res, y_res)
    
    # 6. Evaluate & Interpret [cite: 563, 663]
    evaluation.evaluate_detection(model, X_res, y_res)
    interpretability.explain_model(model, X)

if __name__ == "__main__":
    run_pipeline()
