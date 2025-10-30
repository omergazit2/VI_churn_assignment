import pandas as pd

def feature_extract_web_visits(path = 'data/web_visits.csv'):
    """
    Extract features from web visits data.

    Parameters:
    path (str): Path to the web visits CSV file.

    Returns:
    pd.DataFrame: DataFrame with extracted features.
    """
    
    web_visits = pd.read_csv(path)
    
    diet_titles = [
        'High-fiber meals', 
        'Cholesterol friendly foods', 
        'Mediterranean diet',
        'Healthy eating guide', 
        'Weight management'
    ]

    physical_activity_titles = [
        'Aerobic exercise',
        'Exercise routines', 
        'Strength training basics', 
        'Cardio workouts'
    ]

    sleep_health_titles = [
        'Restorative sleep tips', 
        'Sleep hygiene'
    ]

    resilience_wellbeing_titles = [
        'Stress reduction',
        'Meditation guide'
    ]

    clinical_titles = [
        'Diabetes management',
        'Hypertension basics',
        'Lowering blood pressure',
        'Cardiometabolic health',
        'HbA1c targets'
    ]

    # Create category columns
    web_visits['is_diet'] = web_visits['title'].isin(diet_titles)
    web_visits['is_physical_activity'] = web_visits['title'].isin(physical_activity_titles)
    web_visits['is_sleep'] = web_visits['title'].isin(sleep_health_titles)
    web_visits['is_resilience'] = web_visits['title'].isin(resilience_wellbeing_titles)
    web_visits['is_clinical'] = web_visits['title'].isin(clinical_titles)

    # if one of the category columns is True, is_health_related is True

    web_visits['is_health_related'] = web_visits[['is_diet', 'is_physical_activity', 'is_sleep', 'is_resilience', 'is_clinical']].any(axis=1)
    
    # Aggregate by member
    web_features = web_visits.groupby('member_id').agg({
        'is_diet': 'sum',
        'is_physical_activity': 'sum',
        'is_sleep': 'sum',
        'is_resilience': 'sum',
        'is_clinical': 'sum',
        'is_health_related': 'sum',
        'member_id': 'count'  # Count total visits
    }).rename(columns={
        'is_diet': 'diet_visits',
        'is_physical_activity': 'physical_activity_visits',
        'is_sleep': 'sleep_visits',
        'is_resilience': 'resilience_visits',
        'is_clinical': 'clinical_visits',
        'is_health_related': 'total_health_visits',
        'member_id': 'total_visits'
    }).reset_index()
    
    # Convert counts to ratios (except total_visits)
    ratio_columns = [
        'diet_visits', 
        'physical_activity_visits', 
        'sleep_visits', 
        'resilience_visits', 
        'clinical_visits', 
        'total_health_visits'
    ]
    
    for col in ratio_columns:
        web_features[f"{col}_ratio"] = web_features[col] / web_features['total_visits']
    

    # Calculate non-relevant visits and ratio
    web_features['non_relevant_visits'] = (
        web_features['total_visits'] - web_features['total_health_visits']
    )

    web_features['health_engagement_ratio'] = (
        web_features['total_health_visits'] / web_features['total_visits']
    )

    # Add category diversity
    web_features['category_diversity'] = (
        (web_features['diet_visits'] > 0).astype(int) +
        (web_features['physical_activity_visits'] > 0).astype(int) +
        (web_features['sleep_visits'] > 0).astype(int) +
        (web_features['resilience_visits'] > 0).astype(int) +
        (web_features['clinical_visits'] > 0).astype(int)
    )

    return web_features

def feature_extract_app_usage(path='data/app_usage.csv'):
    """
    Extract features from app usage data.
    
    parameters: 
    path (str): Path to the app usage CSV file.
    
    Returns:
    pd.DataFrame: DataFrame with extracted features.
    """
    app_usage = pd.read_csv(path)
    
    # drop "event_type" column from app_usage DataFrame (all values are the same)
    app_usage.drop(columns=['event_type'], inplace=True)

    app_usage_statistics = app_usage.assign(
        timestamp = pd.to_datetime(app_usage['timestamp'])
    ).groupby('member_id').agg(
        app_usage = ('timestamp', 'count')
    ).reset_index()

    return app_usage_statistics

def feature_extract_claims(path='data/claims.csv'):
    """
    Extract features from claims data.
    
    parameters:
    path (str): Path to the claims CSV file.
    
    Returns:
    pd.DataFrame: DataFrame with extracted features.
    """
    claims = pd.read_csv(path)
    
    # One-hot encode the icd_code column and aggregate by member_id
    claims_dummies = pd.get_dummies(claims, columns=['icd_code'], dtype='int')
    claims_features = claims_dummies.groupby('member_id').agg({
        # Sum all the icd_code dummy columns
        **{col: 'sum' for col in claims_dummies.columns if col.startswith('icd_code_')}
    }).reset_index()

    # count the number of unique icd_codes per member_id
    icd_columns = [col for col in claims_features.columns if col.startswith('icd_code_')]
    claims_features['code_count'] = claims_features[icd_columns].sum(axis=1)

    # count priority conditions
    priority_icds = ['icd_code_E11.9', 'icd_code_I10', 'icd_code_Z71.3']
    priority_cols = [col for col in priority_icds if col in claims_features.columns]
    claims_features['priority_condition_count'] = claims_features[priority_cols].sum(axis=1)
    return claims_features
    
def feature_extract_churn_labels(path='data/churn_labels.csv'):
    """
    Extract churn labels from churn labels data.
    
    parameters:
    path (str): Path to the churn labels CSV file.
    
    Returns:
    pd.DataFrame: DataFrame with member_id and churn_label.
    """
    churn_labels = pd.read_csv(path)
    
    # Define the reference date (May 2025)
    reference_date = pd.to_datetime('2025-05-01')

    # Calculate months in app
    signup_dates = pd.to_datetime(churn_labels['signup_date'])
    churn_labels['months_in_app'] = ((reference_date.year - signup_dates.dt.year) * 12 + 
                                    (reference_date.month - signup_dates.dt.month))

    return churn_labels

def freture_extract_All(path_web ='data/web_visits.csv', path_app='data/app_usage.csv', path_claims='data/claims.csv', path_churn='data/churn_labels.csv'):
    web_features = feature_extract_web_visits(path = path_web)
    app_usage_features = feature_extract_app_usage(path = path_app)
    claims_features = feature_extract_claims(path = path_claims)
    churn_labels = feature_extract_churn_labels(path = path_churn)
    
    # Merge all features on member_id
    all_features = churn_labels.merge(web_features, on='member_id', how='left') \
            .merge(app_usage_features, on='member_id', how='left') \
            .merge(claims_features, on='member_id', how='left') \
            .fillna(0)  
    
    all_features.drop(columns=['signup_date'], inplace=True)
    
    return all_features

if __name__ == "__main__":
    # web_features = feature_extract_web_visits()
    # print(web_features.head())
    
    # app_usage_features = feature_extract_app_usage()
    # print(app_usage_features.head())
    
    # claims_features = feature_extract_claims()
    # print(claims_features.head())
    
    # churn_labels = feature_extract_churn_labels()
    # print(churn_labels.head())
    
    all_features = freture_extract_All()
    print(all_features.head())
    