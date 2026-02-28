def calculate_iwa_metrics(df):
    """
    Quantifies NRW as: System Input Volume - Billed Authorized Consumption.
    Ref: [cite: 547, 628]
    """
    df['nrw_volume'] = df['system_input_volume'] - df['billed_authorized_consumption']
    df['nrw_percent'] = (df['nrw_volume'] / df['system_input_volume']) * 100
    
    # Labeling based on sector benchmark of 25% [cite: 1185]
    df['target'] = (df['nrw_percent'] > 25).astype(int)
    
    # Feature indicators for Physical vs Commercial [cite: 359, 642]
    df['pressure_variability'] = df['pressure'].rolling(window=3).std()
    df['billing_efficiency'] = df['billed_authorized_consumption'] / df['system_input_volume']
    
    return df
